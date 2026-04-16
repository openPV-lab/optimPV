"""Casmopolitan generation node for Ax (categorical & mixed search spaces)."""

# Adapted by @VMLC-PV and @FilipFekete 
# The original work on the Casmopolitan implementation is from the paper: 
# @article{wan2021think,
#   title={Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces},
#   author={Wan, Xingchen and Nguyen, Vu and Ha, Huong and Ru, Binxin and Lu, Cong and Osborne, Michael A},
#   journal={International Conference on Machine Learning (ICML) 38},
#   year={2021}
# }
# The original code is available at: 
# https://github.com/xingchenwan/Casmopolitan

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, override

import numpy as np
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.parameter import ChoiceParameter, RangeParameter
from ax.core.trial_status import TrialStatus
from ax.core.types import TParameterization
from ax.generation_strategy.external_generation_node import ExternalGenerationNode

from optimpv.optimizers.axBOtorch.casmopolitan.optimizer import Optimizer
from optimpv.optimizers.axBOtorch.casmopolitan.optimizer_mixed import MixedOptimizer

"""The algorithm logic is offloaded to the original Casmopolitan implementation:
 - categorical trust region logic: casmopolitan/localbo_cat.py
 - mixed trust region logic: casmopolitan/localbo_mixed.py
 - GP fitting and acqf helpers: casmopolitan/localbo_utils.py
 - suggest/observe/restart orchestration: casmopolitan/optimizer.py, casmopolitan/optimizer_mixed.py
"""

@dataclass
class CasmopolitanNodeState:
    """Runtime state owned by the Casmopolitan node.

    Attributes
    ----------
    optimizer : Optimizer | MixedOptimizer | None
        Casmopolitan optimizer object for categorical or mixed search spaces.
    seen_trial_indices : set[int]
        Completed Ax trial indices already passed to Casmopolitan.
    candidate_queue : list[dict[str, Any]]
        Queue of generated candidates waiting to be returned to Ax.
    """

    optimizer: Optional[Optimizer | MixedOptimizer] = None
    seen_trial_indices: set[int] = field(default_factory=set) # for tracking which completed ax trials have been passed to Casmopolitan
    candidate_queue: list[dict[str, Any]] = field(default_factory=list)


class CasmopolitanGenerationNode(ExternalGenerationNode):
    """ExternalGenerationNode that proposes categorical or mixed candidates with Casmopolitan.

    Parameters
    ----------
    model_options : dict[str, Any] | None, optional
        Configuration options forwarded to the Casmopolitan implementation, by default None.
    batch_size : int, optional
        Number of candidates generated per call to the optimizer, by default 1.
    name : str, optional
        Node name used in the Ax generation strategy, by default ``"CasmopolitanGenerationNode"``.
    maximize : bool, optional
        Whether the Ax objective is maximized or minimized, by default True.
    """
    def __init__(
        self,
        model_options: Optional[dict[str, Any]] = None,
        batch_size: int = 1,
        *,
        name: str = "CasmopolitanGenerationNode",
        maximize: bool = True,
    ) -> None:
        
        super().__init__(name=name)

        self.model_options = dict(model_options or {})
        self.batch_size = int(batch_size)
        self.maximize = bool(maximize)
        self.acq = str(self.model_options.get("acq", "thompson")).lower()
        if self.acq == "ts":
            self.acq = "thompson"
        if self.acq not in {"thompson", "ei", "ucb"}:
            raise ValueError(
                "CasmopolitanGenerationNode supports only acq values 'thompson', 'ei', or 'ucb'."
            )

        # casmopolitans specific kwargs
        self.use_ard = bool(self.model_options.get("use_ard", True))
        self.guided_restart = bool(self.model_options.get("guided_restart", True))
        self.kernel_type = self.model_options.get("kernel_type")
        self.n_cand = self.model_options.get("n_cand")
        self.failtol = self.model_options.get("failtol")
        self.succtol = self.model_options.get("succtol")
        self.length_min_discrete = self.model_options.get("length_min_discrete")
        self.length_max_discrete = self.model_options.get("length_max_discrete")
        self.length_init_discrete = self.model_options.get("length_init_discrete")
        self.length_min = self.model_options.get("length_min")
        self.length_max = self.model_options.get("length_max")
        self.length_init = self.model_options.get("length_init")
        self.tr_multiplier = self.model_options.get("multiplier")
        self.noise_variance = self.model_options.get("noise_variance")
        self.n_training_steps = self.model_options.get("n_training_steps")
        self.max_cholesky_size = self.model_options.get("max_cholesky_size")
        self.min_cuda = self.model_options.get("min_cuda")
        self.param_key_precision = int(self.model_options.get("param_key_precision", 12))

        torch_device = self.model_options.get("torch_device")
        if torch_device is None:
            self.device = "cpu"
        elif isinstance(torch_device, torch.device):
            self.device = torch_device.type
        else:
            self.device = str(torch_device)

        torch_dtype = self.model_options.get("torch_dtype")
        if torch_dtype is None:
            self.dtype = "float32"
        elif torch_dtype == torch.double or torch_dtype == torch.float64:
            self.dtype = "float64"
        else:
            self.dtype = "float32"

        self.state = CasmopolitanNodeState()

        # built later from the Ax experiment search space
        self.param_names: list[str] = []
        self.param_kinds_by_name: dict[str, Literal["categorical", "continuous"]] = {}
        self.category_values_by_name: dict[str, list[Any]] = {}
        self.value_to_index_by_name: dict[str, dict[Any, int]] = {}
        self.lower_by_name: dict[str, float] = {}
        self.upper_by_name: dict[str, float] = {}
        self.continuous_names: list[str] = []
        self.categorical_names: list[str] = []
        self.cont_dims: np.ndarray = np.zeros(0, dtype=int)
        self.cat_dims: np.ndarray = np.zeros(0, dtype=int)
        self.config: np.ndarray = np.zeros(0, dtype=int) # num of categories for each categorical dim
        self.search_space_mode: Literal["categorical", "mixed"] = "categorical"
        self.metric_name: str | None = None

    # offload point to the Caspomolitan algoirthm
    def _build_optimizer(self) -> Optimizer | MixedOptimizer:
        """Based on the category of input parameters, create the Casmopolitan optimizer mixed/categorical for the cached search space metadata.

        Returns
        -------
        Optimizer | MixedOptimizer
            Instance of the categorical or mixed Casmopolitan optimizer.
        """
         
        kwargs: dict[str, Any] = {
            "acq": self.acq,
            "use_ard": self.use_ard,
            "device": self.device,
            "dtype": self.dtype,
        }
        if self.kernel_type is not None:
            kwargs["kernel_type"] = str(self.kernel_type)
        elif self.search_space_mode == "mixed":
            kwargs["kernel_type"] = "mixed"
        else:
            kwargs["kernel_type"] = "transformed_overlap"
        if self.n_cand is not None:
            kwargs["n_cand"] = int(self.n_cand)
        if self.failtol is not None:
            kwargs["failtol"] = int(self.failtol)
        if self.succtol is not None:
            kwargs["succtol"] = int(self.succtol)
        if self.length_min_discrete is not None:
            kwargs["length_min_discrete"] = int(self.length_min_discrete)
        if self.length_max_discrete is not None:
            kwargs["length_max_discrete"] = int(self.length_max_discrete)
        if self.length_init_discrete is not None:
            kwargs["length_init_discrete"] = int(self.length_init_discrete)
        if self.length_min is not None:
            kwargs["length_min"] = float(self.length_min)
        if self.length_max is not None:
            kwargs["length_max"] = float(self.length_max)
        if self.length_init is not None:
            kwargs["length_init"] = float(self.length_init)
        if self.tr_multiplier is not None:
            kwargs["multiplier"] = float(self.tr_multiplier)
        if self.noise_variance is not None:
            kwargs["noise_variance"] = float(self.noise_variance)
        if self.n_training_steps is not None:
            kwargs["n_training_steps"] = int(self.n_training_steps)
        if self.max_cholesky_size is not None:
            kwargs["max_cholesky_size"] = int(self.max_cholesky_size)
        if self.min_cuda is not None:
            kwargs["min_cuda"] = int(self.min_cuda)

        if self.search_space_mode == "categorical":
            return Optimizer(
                config=self.config,
                wrap_discrete=True,
                guided_restart=self.guided_restart,
                **kwargs,
            )

        lb = np.asarray([self.lower_by_name[name] for name in self.continuous_names], dtype=float)
        ub = np.asarray([self.upper_by_name[name] for name in self.continuous_names], dtype=float)
        return MixedOptimizer(
            config=self.config,
            lb=lb,
            ub=ub,
            cont_dims=self.cont_dims,
            cat_dims=self.cat_dims,
            wrap_discrete=True,
            guided_restart=self.guided_restart,
            **kwargs,
        )

    def _initialize_ax_parameter_cache(self, experiment: Experiment) -> None:
        """Cache Casmopolitan search space metadata from the Ax experiment.

        Parameters
        ----------
        experiment : Experiment
            Ax experiment object from which the search space parameter metadata is pulled.

        Raises
        ------
        NotImplementedError
            If the search space contains unsupported parameter types or is purely continuous.
        ValueError
            If the experiment has no optimization config or does not have exactly one objective metric.
        """
        search_space = experiment.search_space

        self.param_names = []
        self.param_kinds_by_name = {}
        self.category_values_by_name = {}
        self.value_to_index_by_name = {}
        self.lower_by_name = {}
        self.upper_by_name = {}
        self.continuous_names = []
        self.categorical_names = []

        for param in search_space.parameters.values():
            if isinstance(param, ChoiceParameter):
                values = list(param.values)
                self.param_names.append(param.name)
                self.param_kinds_by_name[param.name] = "categorical"
                self.categorical_names.append(param.name)
                self.category_values_by_name[param.name] = values
                self.value_to_index_by_name[param.name] = {value: idx for idx, value in enumerate(values)}
            elif isinstance(param, RangeParameter) and "float" in str(param.parameter_type).lower():
                self.param_names.append(param.name)
                self.param_kinds_by_name[param.name] = "continuous"
                self.continuous_names.append(param.name)
                self.lower_by_name[param.name] = float(param.lower)
                self.upper_by_name[param.name] = float(param.upper)
            else:
                raise NotImplementedError(
                    "CasmopolitanGenerationNode supports ChoiceParameters "
                    "for categorical dimensions and floating RangeParameters for "
                    "continuous mixed-space dimensions."
                )

        self.cat_dims = np.asarray(
            [idx for idx, name in enumerate(self.param_names) if self.param_kinds_by_name[name] == "categorical"],
            dtype=int,
        )
        self.cont_dims = np.asarray(
            [idx for idx, name in enumerate(self.param_names) if self.param_kinds_by_name[name] == "continuous"],
            dtype=int,
        )
        self.config = np.asarray(
            [len(self.category_values_by_name[name]) for name in self.categorical_names],
            dtype=int,
        )

        if len(self.cont_dims) == 0:
            self.search_space_mode = "categorical"
        elif len(self.cat_dims) == 0:
            raise NotImplementedError(
                "CasmopolitanGenerationNode does not support purely continuous spaces. "
                "Use TuRBOGenerationNode for continuous-only optimization."
            )
        else:
            self.search_space_mode = "mixed"

        if experiment.optimization_config is None:
            raise ValueError("Experiment has no optimization_config configured.")
        metric_names = list(experiment.optimization_config.metrics.keys())
        if len(metric_names) != 1:
            raise ValueError("CasmopolitanGenerationNode supports only a single objective metric.")
        self.metric_name = metric_names[0]

        self.state.optimizer = self._build_optimizer()

    def _key(self, params: dict[str, Any]) -> tuple[Any, ...]:
        """Stable key for deduplicating candidates against pending trials."""

        key_vals: list[Any] = []
        for name in self.param_names:
            value = params[name]
            if isinstance(value, float):
                key_vals.append(round(float(value), self.param_key_precision))
            else:
                key_vals.append(value)
        return tuple(key_vals)

    def _params_to_ordinal(self, params: dict[str, Any]) -> np.ndarray:
        """Convert one Ax parameterization into a Casmopolitan input vector.

        Parameters
        ----------
        params : dict[str, Any]
            Candidate parameterization in Ax space.

        Returns
        -------
        np.ndarray
            Numeric input vector expected by the Casmopolitan backend.
        """
        values: list[float] = []
        for name in self.param_names:
            if self.param_kinds_by_name[name] == "categorical":
                values.append(float(self.value_to_index_by_name[name][params[name]]))
            else:
                values.append(float(params[name]))
        return np.asarray(values, dtype=float)

    def _ordinal_to_params(self, x: np.ndarray) -> dict[str, Any]:
        """Convert one Casmopolitan candidate back into an Ax parameterization.

        Parameters
        ----------
        x : np.ndarray
            Casmopolitan candidate vector.

        Returns
        -------
        dict[str, Any]
            Candidate parameterization in Ax space.

        Raises
        ------
        RuntimeError
            If Casmopolitan proposes an out-of-range categorical index.
        """
        params: dict[str, Any] = {}
        for idx, name in enumerate(self.param_names):
            if self.param_kinds_by_name[name] == "categorical":
                ordinal = int(round(float(x[idx])))
                values = self.category_values_by_name[name]
                if ordinal < 0 or ordinal >= len(values):
                    raise RuntimeError(f"Casmopolitan proposed out-of-range category {ordinal} for '{name}'.")
                params[name] = values[ordinal]
            else:
                raw_value = float(x[idx])
                params[name] = min(max(raw_value, self.lower_by_name[name]), self.upper_by_name[name])
        return params

    def _enqueue_batch_from_casmopolitan(self) -> None:
        """Generate a Casmopolitan batch and store it one by one in the queue.

        Raises
        ------
        RuntimeError
            If the backend optimizer has not been initialized.
        """
        optimizer = self.state.optimizer
        if optimizer is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        X_next = optimizer.suggest(self.batch_size)
        for row in X_next:
            self.state.candidate_queue.append(self._ordinal_to_params(row))

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """Read newly completed Ax trials and feed them back to Casmopolitan.

        Parameters
        ----------
        experiment : Experiment
            Ax experiment object containing the search space and trial objects.
        data : Data
            Ax data object containing completed trial results.

        Raises
        ------
        RuntimeError
            If the generator state was not initialized correctly.
        """
        if self.state.optimizer is None:
            self._initialize_ax_parameter_cache(experiment=experiment)
        if self.metric_name is None or self.state.optimizer is None:
            raise RuntimeError("Generator state not initialized correctly.")

        completed_trials = sorted(
            (trial for trial in experiment.trials.values() if trial.status == TrialStatus.COMPLETED),
            key=lambda trial: trial.index,
        )

        X_rows: list[np.ndarray] = []
        Y_rows: list[float] = []
        new_trial_indices: list[int] = []

        for trial in completed_trials:
            if trial.index in self.state.seen_trial_indices:
                continue

            params = trial.arm.parameters  # type: ignore[attr-defined]
            trial_df = data.df[data.df["trial_index"] == trial.index]
            row = trial_df[trial_df["metric_name"] == self.metric_name]
            if row.empty:
                continue

            value = float(row["mean"].iloc[0])
            if not np.isfinite(value):
                continue

            X_rows.append(self._params_to_ordinal(params))
            # The original Casmopolitan implementation assumes minimization. Keep
            # Ax metric values unchanged, but flip maximizing objectives for
            # the internal suggest/observe loop so trust region updates are aligned
            # with the actual optimization direction.
            Y_rows.append(-value if self.maximize else value)
            new_trial_indices.append(trial.index)

        if len(X_rows) == 0:
            return

        if self.state.optimizer.batch_size is None:
            # If Casmopolitan starts after a Sobol phase, completed Ax trials already exist
            # before the wrapper sees its first `suggest(...)` call. Bootstrap the internal
            # batch size here and disable the wrapper's own random init queue so these
            # completed Sobol trials serve as the initial design.
            self.state.optimizer.batch_size = self.batch_size
            self.state.optimizer.casmopolitan.batch_size = self.batch_size
            self.state.optimizer.X_init = np.zeros((0, self.state.optimizer.true_dim))
            self.state.optimizer.casmopolitan._X = np.zeros(
                (0, self.state.optimizer.casmopolitan.dim)
            )
            self.state.optimizer.casmopolitan._fX = np.zeros((0, 1))

        self.state.candidate_queue.clear()
        self.state.optimizer.observe(np.vstack(X_rows), np.asarray(Y_rows)) # this is where the Casompolitan update happens
        self.state.seen_trial_indices.update(new_trial_indices)

    @override
    def get_next_candidate(self, pending_parameters: list[TParameterization]) -> TParameterization:
        """Return one candidate to Ax, skipping duplicates against pending trials.

        Parameters
        ----------
        pending_parameters : list[TParameterization]
            Candidates already proposed by Ax that are still pending.

        Returns
        -------
        TParameterization
            Parameterization for the next candidate to evaluate.

        Raises
        ------
        RuntimeError
            If the optimizer has not been initialized.
        """
        if self.state.optimizer is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        pending_keys: set[tuple[Any, ...]] = set()
        for pending in pending_parameters:
            try:
                pending_keys.add(self._key(dict(pending)))
            except Exception:
                continue

        draws = 0
        max_draws = max(10, 5 * self.batch_size)
        while True:
            if len(self.state.candidate_queue) == 0:
                self._enqueue_batch_from_casmopolitan()

            params = self.state.candidate_queue.pop(0)
            key = self._key(params)
            if key in pending_keys and draws < max_draws:
                draws += 1
                continue
            return params
