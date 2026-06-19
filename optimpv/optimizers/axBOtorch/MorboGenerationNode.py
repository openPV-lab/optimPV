"""Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces (morbo) generation node for use in Ax."""
# Adapted by @VMLC-PV and @FilipFekete 
# The original work on the morbo implementation is from the paper: 
#
# @InProceedings{pmlr-v162-daulton22a,
# title = 	 {Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces},
# author =       {Daulton, Samuel and Eriksson, David and Balandat, Maximilian and Bakshy, Eytan},
# booktitle = 	 {Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence},
# year = 	 {2022},
# series = 	 {Proceedings of Machine Learning Research},
# publisher =    {PMLR},
# }
#
# And this code is adapted from the original MORBO, which is available at
# https://github.com/facebookresearch/morbo
# Please consider citing the original paper if you use this code in your research. 

import copy
from dataclasses import dataclass, field # field for mutable default across instances, gives each instance its own default value
from typing import Any, Optional, override

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.parameter import RangeParameter
from ax.core.trial_status import TrialStatus
from ax.core.types import TParameterization
from ax.exceptions.core import DataRequiredError
from ax.generation_strategy.external_generation_node import ExternalGenerationNode

# MORBO internals
from optimpv.optimizers.axBOtorch.morbo.gen import TS_select_batch_MORBO
from optimpv.optimizers.axBOtorch.morbo.state import TRBOState
from optimpv.optimizers.axBOtorch.morbo.trust_region import TurboHParams

# MORBO state wraps a delegated engine object
@dataclass
class MorboNodeState:
    """Runtime state owned by the generation node.

    Attributes
    ----------
    trbo_state : TRBOState | None
        Current TRBO engine state.
    seen_trial_indices : set[int]
        Completed Ax trial indices already processed by the node.
    pending_tr_index_by_key : dict[tuple[Any, ...], int]
        Mapping from candidate keys to proposing trust-region indices for pending Ax trials.
    candidate_queue : list[dict[str, Any]]
        Queue of generated candidates waiting to be returned to Ax.
    """

    trbo_state: Optional[TRBOState] = None # current MORBO/TRBO engine state
    seen_trial_indices: set[int] = field(default_factory=set) # completed trials
    pending_tr_index_by_key: dict[tuple[Any, ...], int] = field(default_factory=dict) # map candidate key -> proposing TR index
    candidate_queue: list[dict[str, Any]] = field(default_factory=list) # generated candidates waiting 

class MorboGenerationNode(ExternalGenerationNode):
    """ExternalGenerationNode that proposes points using MORBO/TRBO, with focus on multiobjective optimization.

    Space convention:
    - This node runs directly in Ax parameter space (no FitParam descaling here).
    - MORBO internally normalizes using the bounds we pass to TRBOState.
    Parameters
    ----------
    model_options : dict[str, Any] | None, optional
        Options controlling MORBO state construction, objective handling, and
        trust-region hyperparameters, by default None.
    batch_size : int, optional
        Number of candidates proposed per generation step, by default 1.
    device : torch.device | None, optional
        Torch device used for tensors and MORBO internals, by default None.
    dtype : torch.dtype | None, optional
        Torch dtype used for tensors and MORBO internals, by default None.
    name : str, optional
        Node name used in the Ax generation strategy, by default ``"MorboGenerationNode"``.
    """
    def __init__( 
        self,
        model_options: Optional[dict[str, Any]] = None,
        batch_size: int = 1,
        *, # keyword-only args below
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        name: str = "MorboGenerationNode",
    ) -> None:
        
        super().__init__(name=name) # call parent class contructor to run base-class setup and register the node name

        self.model_options = dict(model_options or {})

        self.metric_names = None
        self.minimize_flags = None
        self.batch_size = int(batch_size)
        self.reference_point = self.model_options.get("reference_point")
        self.max_evals = int(self.model_options.get("max_evals", 200))
        self.n_initial_points = int(self.model_options.get("n_initial_points", 20))
        self.tr_hparam_overrides = self.model_options.get("tr_hparam_overrides", {}) or {}
        self.param_key_precision = int(self.model_options.get("param_key_precision", 12))
        self.verbose = bool(self.model_options.get("verbose", False))

        if device is None and "torch_device" in self.model_options:
            device = self.model_options["torch_device"]
        if dtype is None and "torch_dtype" in self.model_options:
            dtype = self.model_options["torch_dtype"]
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.double

        self.num_objectives = 0
        self.state = MorboNodeState()

        # empty declaration, later initialized from Ax experiment search space on first fit.
        self.parameters: list[RangeParameter] | None = None
        self.param_names: list[str] = []
        self.int_param_names: set[str] = set()
        self.bounds_tensor: torch.Tensor | None = None

    # safer method to pull objective metadata from Ax experiment, with fallback logic and warnings
    def _get_objectives_from_experiment(self, experiment: Experiment) -> tuple[list[str], list[bool]]:
        """Get objective metric names + directions from Ax optimization config.

        Parameters
        ----------
        experiment : Experiment
            Ax experiment object containing the optimization configuration.

        Returns
        -------
        tuple[list[str], list[bool]]
            Objective metric names and metric minimization flags.

        Raises
        ------
        ValueError
            If the experiment has no optimization config or no optimization metrics.
        """
        # Pulls per-metric direction from Ax
        if experiment.optimization_config is None:
            raise ValueError("Experiment has no optimization_config configured.")

        # metric_names = list(experiment.optimization_config.metrics.keys())
        metric_names = experiment.optimization_config._objective.metric_names
        if len(metric_names) == 0:
            raise ValueError("No optimization metrics found in experiment configuration.")

        minimize_by_name: dict[str, bool] = {}
        objective = experiment.optimization_config.objective

        # Multi-objective shape: avoid touching `.metric`, which raises on MultiObjective.
        objective_list = None
        try:
            objective_list = list(getattr(objective, "objectives", []) or [])
        except Exception:
            objective_list = None

        if objective_list:
            for obj in objective_list:
                obj_metric = getattr(obj, "metric", None)
                if obj_metric is not None:
                    minimize_by_name[obj_metric.name] = bool(getattr(obj, "minimize", False))
        else:
            # single-objective shape
            # Ax could still pass a single-objective optimization_config, just a fallback to not crash, raises a warning later 
            metric = getattr(objective, "metric", None)
            if metric is not None:
                minimize_by_name[metric.name] = bool(getattr(objective, "minimize", False))

        minimize_flags: list[bool] = []
        for name in metric_names:
            if name in minimize_by_name:
                minimize_flags.append(minimize_by_name[name])
            else:
                # metric_obj = experiment.optimization_config.metrics[name]
                minimize_flags.append(bool(getattr(name, "lower_is_better", False)))

        return metric_names, minimize_flags

    def _ensure_objectives(self, experiment: Experiment) -> None:
        """Resolve objective metadata from model_options or Ax config.

        Parameters
        ----------
        experiment : Experiment
            Ax experiment object containing the optimization configuration.

        Raises
        ------
        ValueError
            If objective metadata is empty or inconsistent.
        """
        if self.metric_names is None or self.minimize_flags is None:
            self.metric_names, self.minimize_flags = self._get_objectives_from_experiment(experiment)

        # validates lengths and sets num_objectives
        self.metric_names = list(self.metric_names)
        self.minimize_flags = [bool(x) for x in self.minimize_flags]
        if len(self.metric_names) == 0:
            raise ValueError("metric_names must contain at least one objective.")
        if len(self.metric_names) != len(self.minimize_flags):
            raise ValueError("metric_names and minimize_flags must have the same length.")
        self.num_objectives = len(self.metric_names)
        if self.num_objectives == 1:
            print(
                "MorboGenerationNodeScratch is intended for multi-objective optimization; "
                "single-objective runs may be better served by TuRBO."
            )

    def _initialize_ax_parameter_cache(self, experiment: Experiment) -> None:
        """Cache parameter metadata from Ax experiment creating optimizer space.
           Reads the created Ax search space and stores what MORBO needs (parameter names, bounds)
           in a form of tensor for later use in TRBOState.

        Parameters
        ----------
        experiment : Experiment
            Ax experiment object its search space defines the parameter metadata.

        Raises
        ------
        NotImplementedError
            If the search space contains parameters other than
            ``RangeParameter`` instances.
        """
        search_space = experiment.search_space
        if any(not isinstance(p, RangeParameter) for p in search_space.parameters.values()):
            raise NotImplementedError("MorboGenerationNodeScratch supports only RangeParameters.")

        self.parameters = list(search_space.parameters.values())  # type: ignore[arg-type]
        self.param_names = [p.name for p in self.parameters] # cache in order that defines the order of columns in candidate tensors
        self.int_param_names = { # detect integer parameters
            p.name for p in self.parameters if "int" in str(p.parameter_type).lower()
        }
        lb = [float(p.lower) for p in self.parameters]
        ub = [float(p.upper) for p in self.parameters]
        self.bounds_tensor = torch.tensor([lb, ub], dtype=self.dtype, device=self.device)

    def _key(self, params: dict[str, Any]) -> tuple[Any, ...]:
        """Stable key for matching proposed params to completed trial callbacks.
           Store which trust region index generated a pending candidate and recover that mapping when Ax returns completed trial data.
        """
        key_vals: list[Any] = []
        for name in self.param_names:
            val = params[name]
            if name in self.int_param_names:
                key_vals.append(int(val))
            else:
                key_vals.append(round(float(val), self.param_key_precision))
        return tuple(key_vals)
    
    def _build_tr_hparams(self) -> TurboHParams:
        """Build MORBO trust-region hyperparameters for TRBOState.

        Returns
        -------
        TurboHParams
            MORBO trust region hyperparameters passed to ``TRBOState``.

        Raises
        ------
        RuntimeError
            If objective directions are not initialized.
        ValueError
            If the configured reference point has the wrong dimension.
        """
        if self.minimize_flags is None:
            raise RuntimeError("Objective directions are not initialized.")
        tr_kwargs = copy.deepcopy(self.tr_hparam_overrides)
        for k in ("batch_size", "n_initial_points", "max_reference_point", "verbose"):
            tr_kwargs.pop(k, None)
        tr_kwargs.setdefault("hypervolume", self.num_objectives > 1)
        tr_kwargs.setdefault("failure_streak", max(len(self.param_names) // 3, 10))
        tr_kwargs.setdefault("min_tr_size", self.n_initial_points)

        # flips reference_point based on minimize flags
        signed_ref_point = None
        if self.reference_point is not None:
            if len(self.reference_point) != self.num_objectives:
                raise ValueError("reference_point length must match number of objectives.")
            signed_ref_point = []
            for rp, minimize in zip(self.reference_point, self.minimize_flags):
                signed_val = (-1.0 if minimize else 1.0) * float(rp)
                signed_ref_point.append(signed_val)

        return TurboHParams(
            batch_size=self.batch_size,
            n_initial_points=self.n_initial_points,
            max_reference_point=signed_ref_point,
            verbose=self.verbose,
            **tr_kwargs,
        )
    
    def _initialize_trbo(
        self, X_init: torch.Tensor, Y_init: torch.Tensor, tr_indices: torch.Tensor
    ) -> None:
        """Create initial TRBO state from completed Sobol/Ax trials.

        Parameters
        ----------
        X_init : torch.Tensor
            Initial observed points in Ax parameter space.
        Y_init : torch.Tensor
            Initial objective values oriented to MORBO's maximize convention.
        tr_indices : torch.Tensor
            Trust-region indices assigned to the initial observations.

        Raises
        ------
        RuntimeError
            If bounds have not been initialized.
        ValueError
            If the trust-region hyperparameters are inconsistent.
        """
        if self.bounds_tensor is None:
            raise RuntimeError("Bounds not initialized.")
        # build TRBO hyperparamters, class from MORBO implementation
        tr_hparams = self._build_tr_hparams()
        if tr_hparams.n_initial_points < tr_hparams.min_tr_size:
            raise ValueError("n_initial_points must be >= min_tr_size.")

        # Offload MORBO state/model/region management to original MORBO implementation.
        # We pass the initial data and trust region indices from completed Sobol trials, and let MORBO handle restarts and region updates from there.
        # Handoff to Morbo internal state management (morbo/state.py)
        trbo_state = TRBOState(
            dim=len(self.param_names),
            max_evals=self.max_evals,
            num_outputs=self.num_objectives,
            num_objectives=self.num_objectives,
            bounds=self.bounds_tensor,
            tr_hparams=tr_hparams,
            constraints=None,
            objective=None,
        )
        trbo_state.update(X=X_init, Y=Y_init, new_ind=tr_indices) # pass initial observed data
        trbo_state.log_restart_points(X=X_init, Y=Y_init) # record these points as reference points for TRBO restart logic
        for tr_idx in range(tr_hparams.n_trust_regions):
            trbo_state.initialize_standard(
                tr_idx=tr_idx,
                restart=False,
                switch_strategy=False,
                X_init=X_init,
                Y_init=Y_init,
            )
        trbo_state.update_data_across_trs() # synchornize data across trust regions
        trbo_state.TR_index_history.fill_(-2)
        self.state.trbo_state = trbo_state

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """Pull completed Ax trials, objectives, and update TRBO state.

        Parameters
        ----------
        experiment : Experiment
            Ax experiment object containing the search space and trial objects.
        data : Data
            Ax data object containing completed trial results.

        Raises
        ------
        DataRequiredError
            If no completed trials are available yet.
        RuntimeError
            If objective metadata cannot be resolved.
        """
        if self.parameters is None:
            self._initialize_ax_parameter_cache(experiment=experiment)
        self._ensure_objectives(experiment=experiment)
        if self.metric_names is None or self.minimize_flags is None:
            raise RuntimeError("Objective metadata not initialized.")

        completed_trials = sorted(
            (t for t in experiment.trials.values() if t.status == TrialStatus.COMPLETED),
            key=lambda t: t.index,
        )
        if len(completed_trials) == 0:
            raise DataRequiredError("No completed trials available yet for MORBO.")

        # for newly completed trials and data
        X_rows: list[torch.Tensor] = []
        Y_rows: list[torch.Tensor] = []
        tr_indices: list[int] = []
        new_trial_indices: list[int] = []

        # read new completed trials, skip seen trials
        for trial in completed_trials:
            if trial.index in self.state.seen_trial_indices:
                continue

            params = trial.arm.parameters  # type: ignore[attr-defined]
            trial_df = data.df[data.df["trial_index"] == trial.index]

            y_vals: list[float] = []
            missing_metric = False
            for metric_idx in range(len(self.metric_names)):
                metric_name = self.metric_names[metric_idx]
                minimize = self.minimize_flags[metric_idx]
                row = trial_df[trial_df["metric_name"] == metric_name]
                if row.empty:
                    missing_metric = True
                    break
                val = float(row["mean"].iloc[0])
                # Keep node objective in a maximize-oriented space.
                # Flipping the sign for minimization metrics
                y_vals.append(-val if minimize else val)
            if missing_metric:
                continue

            x_vals = [float(params[name]) for name in self.param_names]
            X_rows.append(torch.tensor(x_vals, dtype=self.dtype, device=self.device))
            Y_rows.append(torch.tensor(y_vals, dtype=self.dtype, device=self.device))
            new_trial_indices.append(trial.index)
            tr_indices.append(self.state.pending_tr_index_by_key.pop(self._key(params), 0))

        if len(X_rows) == 0:
            return

        self.state.candidate_queue.clear()
        X_new = torch.stack(X_rows) # build from trial parameters (Ax space)
        Y_new = torch.stack(Y_rows) # build from trial metrics, sign-oriented 
        tr_idx_tensor = torch.tensor(tr_indices, dtype=torch.long, device=self.device)

        # first batch initializes TRBO state; subsequent batches update state and potentially trigger restarts
        if self.state.trbo_state is None:
            self._initialize_trbo(X_init=X_new, Y_init=Y_new, tr_indices=tr_idx_tensor)
        else:
            # Offload trust-region update/restart logic to MORBO state implementation.
            trbo_state = self.state.trbo_state
            trbo_state.update(X=X_new, Y=Y_new, new_ind=tr_idx_tensor)
            should_restart = trbo_state.update_trust_regions_and_log(
                X_cand=X_new,
                Y_cand=Y_new,
                tr_indices=tr_idx_tensor,
                batch_size=X_new.shape[0],
                verbose=False,
            )
            switch_strategy = trbo_state.check_switch_strategy()
            if switch_strategy:
                should_restart = [True for _ in should_restart]
            if any(should_restart):
                for tr_idx in range(trbo_state.tr_hparams.n_trust_regions):
                    if should_restart[tr_idx]:
                        trbo_state.initialize_standard(
                            tr_idx=tr_idx,
                            restart=True,
                            switch_strategy=switch_strategy,
                        )
            trbo_state.update_data_across_trs()

        self.state.seen_trial_indices.update(new_trial_indices)

    def _enqueue_batch_from_morbo(self) -> None:
        """Get a new batch of candidates from MORBO and store them in FIFO queue.

        Raises
        ------
        RuntimeError
            If the MORBO / TRBO state has not been initialized.
        """
        trbo_state = self.state.trbo_state # pull current state
        if trbo_state is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        # Candidate generation itself is delegated to original MORBO logic.
        selection = TS_select_batch_MORBO(trbo_state=trbo_state) # returns a batch of candidate points (tensor) and their TR indices 
        trbo_state.tabu_set.log_iteration()
      
        for cand_idx in range(len(selection.X_cand)):
            x = selection.X_cand[cand_idx]
            tr_idx = selection.tr_indices[cand_idx]
            x_vals = x.tolist()
            raw: dict[str, float] = {}
            for param_idx in range(len(self.param_names)):
                name = self.param_names[param_idx]
                raw[name] = float(x_vals[param_idx])
            for name in self.int_param_names:
                raw[name] = int(round(raw[name]))
            self.state.candidate_queue.append({"params": raw, "tr_idx": int(tr_idx)}) # push cancidtes into internal FIFO queue with AX params and TR index

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
            If the MORBO / TRBO state has not been initialized.
        """
        # ensure state exists
        if self.state.trbo_state is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        # key set of pending params
        pending_keys: set[tuple[Any, ...]] = set()
        for p in pending_parameters:
            try:
                pending_keys.add(self._key(dict(p)))
            except Exception:
                continue

        draws = 0 # counter for the amount of skipped duplicate candidates
        max_draws = max(10, 5 * self.batch_size)
        while True:
            if len(self.state.candidate_queue) == 0:
                self._enqueue_batch_from_morbo()
            item = self.state.candidate_queue.pop(0)
            params = item["params"]
            tr_idx = item["tr_idx"]
            key = self._key(params)
            if key in pending_keys and draws < max_draws:
                draws += 1
                continue
            self.state.pending_tr_index_by_key[key] = tr_idx
            return params
        
