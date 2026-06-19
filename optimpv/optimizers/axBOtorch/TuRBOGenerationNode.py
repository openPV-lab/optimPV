"""Trust Region Bayesian Optimization (TuRBO) generation node for us in Ax."""

# Co-authored-by: Rhys Goodall <rhys.goodall@outlook.com> 
# Adapted by @VMLC-PV and @FilipFekete from the original code by Rhys Goodall, 
# see original code and discussion in the Ax repository:
# https://github.com/facebook/Ax/issues/4801
#
# The original work on the TuRBO implementation is from the paper: 
#
# @inproceedings{eriksson2019scalable,
#   title = {Scalable Global Optimization via Local {Bayesian} Optimization},
#   author = {Eriksson, David and Pearce, Michael and Gardner, Jacob and Turner, Ryan D and Poloczek, Matthias},
#   booktitle = {Advances in Neural Information Processing Systems},
#   pages = {5496--5507},
#   year = {2019},
#   url = {http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization.pdf},
# }
# Please consider citing the original paper if you use this code in your research. 
# 
# The BoTorch tutorial on TuRBO was also a helpful resource in implementing this generation node: https://botorch.org/docs/tutorials/turbo_1/

import math
from typing import Any, Literal, Self, override

import torch
import gpytorch
from ax.core.data import Data
from ax.api.client import Client
from ax.core.experiment import Experiment
from ax.core.parameter import RangeParameter
from ax.core.trial_status import TrialStatus
from ax.core.types import TParameterization
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.fit import fit_gpytorch_mll  
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf 
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from pydantic import BaseModel
from torch.quasirandom import SobolEngine



def fit_gp(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    normalize_inputs: bool = True,
    standardize_outputs: bool = True,
    max_cholesky_size: float = float("inf"),  # pyright: ignore[reportCallInDefaultInitializer]
) -> SingleTaskGP:
    """Build and fit a single-task Gaussian process model.

    Parameters
    ----------
    X : torch.Tensor
        Training inputs of shape ``(n, d)``.
    Y : torch.Tensor
        Training targets of shape ``(n, 1)``.
    normalize_inputs : bool, optional
        Whether to normalize the inputs to [0, 1]^d,
        by default True.
    standardize_outputs : bool, optional
        Whether to standardize the outputs zero mean, unit variance, by default True.
    max_cholesky_size : float, optional
        Maximum size for Cholesky decomposition,
        by default ``float("inf")``.

    Returns
    -------
    SingleTaskGP
        Fitted BoTorch single-task GP model.
    """

    d = X.shape[-1]
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    input_transform = Normalize(d=d) if normalize_inputs else None
    outcome_transform = Standardize(m=1) if standardize_outputs else None

    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_constraint=Interval(0.005, 4.0)))

    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
        likelihood=likelihood,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)


    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        fit_gpytorch_mll(mll)  # pyright: ignore[reportUnusedCallResult]


    return model

class TurboState(BaseModel):
    """State of the TuRBO algorithm.

    Attributes
    ----------
    dim : int
        Number of search-space dimensions.
    batch_size : int
        Number of candidates generated per TuRBO iteration.
    length : float
        Current trust-region side length in normalized space.
    length_min : float
        Minimum trust-region length before a restart is triggered.
    length_max : float
        Maximum allowable trust-region length.
    failure_counter : int
        Number of consecutive non-improving updates.
    success_counter : int
        Number of consecutive improving updates.
    success_tolerance : int
        Number of improving updates required to expand the trust region.
    failure_tolerance : float | None
        Number of non-improving updates required to shrink the trust region.
    best_value : float | None
        Best objective value seen so far.
    restart_triggered : bool
        Whether the trust region has shrunk below the minimum threshold.
    maximize : bool
        Whether the objective is being maximized or minimized.
    """

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 10
    failure_tolerance: float | None = None
    best_value: float | None = None
    restart_triggered: bool = False
    maximize: bool = True  # Whether we are maximizing or minimizing the objective

    def model_post_init(self, __context: Any) -> None:  # noqa: PYI063
        """Initialize optional fields based on the required fields.

        Parameters
        ----------
        __context : Any
            Optional Pydantic initialization context.
        """
        if self.best_value is None:
            self.best_value = -float("inf") if self.maximize else float("inf")

        if self.failure_tolerance is None:
            self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))

    def update_state(self, Y_next: torch.Tensor) -> Self:
        """Update the state of the TuRBO algorithm. If the new set of points
        has an improvement over the best value, we increase the success counter.
        If the new set of points does not have an improvement, we increase the
        failure counter. The size of the trust region is updated based on the
        running values of the success and failure counters. If the trust region
        becomes too small then we trigger a restart.

        Parameters
        ----------
        Y_next : torch.Tensor
            Objective values from the latest batch of evaluated candidates.

        Returns
        -------
        Self
            Updated TuRBO state.

        Raises
        ------
        RuntimeError
            If ``best_value`` has not been initialized.
        """
        if self.best_value is None:
            raise RuntimeError("Best value not initialized. This is a bug.")

        if self.maximize:
            best_new = Y_next.max().item()
            is_improvement = best_new > self.best_value + 1e-3 * math.fabs(self.best_value)  # pyright: ignore[reportArgumentType, reportOptionalOperand]
        else:
            best_new = Y_next.min().item()
            is_improvement = best_new < self.best_value - 1e-3 * math.fabs(self.best_value)  # pyright: ignore[reportArgumentType, reportOptionalOperand]

        if is_improvement:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0

        if self.maximize:
            self.best_value = max(self.best_value, best_new)
        else:
            self.best_value = min(self.best_value, best_new)

        if self.length < self.length_min:
            self.restart_triggered = True
        return self

    def get_trust_region_bounds(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        buffer: float = 0.0,
        weights: torch.Tensor | None = None,
        maximize: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the trust region bounds for the TuRBO algorithm.

        Parameters
        ----------
        X : torch.Tensor
            Evaluated points in normalized space.
        Y : torch.Tensor
            Objective values corresponding to ``X``.
        buffer : float, optional
            By default 0.0.
        weights : torch.Tensor | None, optional
            Per-dimension scaling weights, typically derived from GP
            lengthscales, by default None.
        maximize : bool | None, optional
            Unused argument, by default None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Lower bound, upper bound, and trust-region center in normalized
            coordinates.

        Raises
        ------
        ValueError
            If provided ``weights`` do not have the same shape as the center
            point.
        """
        x_center = X[Y.argmax() if self.maximize else Y.argmin(), :].clone()
        if weights is None:
            weights = torch.ones_like(x_center)  # Initial weights before model fitting
        else:
            if weights.shape != x_center.shape:
                raise ValueError("Weights must have the same shape as the center point.")
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        rescaling_factor = (1 + buffer) * 2.0
        tr_lb = torch.clamp(x_center - weights * self.length * rescaling_factor, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.length * rescaling_factor, 0.0, 1.0)

        return tr_lb, tr_ub, x_center

    def get_trust_region_data(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        X_pending: torch.Tensor | None = None,
        buffer: float = 0.0,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Get the trust region data for the TuRBO algorithm.

        Parameters
        ----------
        X : torch.Tensor
            Evaluated points in normalized space.
        Y : torch.Tensor
            Objective values corresponding to ``X``.
        X_pending : torch.Tensor | None, optional
            Pending points in normalized space, by default None.
        buffer : float, optional
            by default 0.0.
        weights : torch.Tensor | None, optional
            Per-dimension scaling weights for the trust region, by default None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
            Points, objective values, and pending points from the
            trust region.
        """
        # Get best point and create trust region
        tr_lb, tr_ub, _ = self.get_trust_region_bounds(X, Y, buffer=buffer, weights=weights)  # pyright: ignore[reportArgumentType]

        # Filter points within trust region + buffer for training
        mask = torch.all((tr_lb <= X) & (tr_ub >= X), dim=1)
        X_train = X[mask]
        Y_train = Y[mask]

        if X_pending is not None:
            X_pending = X_pending[mask]

        return X_train, Y_train, X_pending

    def generate_batch(
        self,
        model: SingleTaskGP,  # GP model
        X: torch.Tensor,  # Evaluated points on the domain [0, 1]^d
        Y: torch.Tensor,  # Function values
        batch_size: int,
        X_pending: torch.Tensor | None = None,
        acqf: Literal["ei", "ts"] = "ts",
        acqf_kwargs: dict[str, Any] | None = None,
        # BoTorch inequality format: (indices, coeffs, rhs) for X[:, indices] @ coeffs <= rhs
        inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]] | None = None,
        bounds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a batch of candidates within the current trust region.

        Parameters
        ----------
        model : SingleTaskGP
            Fitted Gaussian process model.
        X : torch.Tensor
            Evaluated points in normalized space.
        Y : torch.Tensor
            Objective values corresponding to ``X``.
        batch_size : int
            Number of candidates to generate.
        X_pending : torch.Tensor | None, optional
            Pending points in normalized space, by default None.
        acqf : Literal["ei", "ts"], optional
            Acquisition function used to generate candidates, by default ``"ts"``.
        acqf_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments for the acquisition function,
            by default None.
        inequality_constraints : list[tuple[torch.Tensor, torch.Tensor, float]] | None, optional
            BoTorch-style inequality constraints, by default None.
        bounds : torch.Tensor | None, optional
            Parameter bounds in physical space, required when applying
            ``inequality_constraints``, by default None.

        Returns
        -------
        torch.Tensor
            Batch of generated candidate points in normalized space.

        Raises
        ------
        ValueError
            If inequality constraints are provided without physical-space
            bounds.
        RuntimeError
            If all Thompson sampling candidates are filtered out by
            constraints.
        """
        if acqf_kwargs is None:
            acqf_kwargs = {}

        assert acqf in ("ts", "ei")
        assert X.min() >= 0.0
        assert X.max() <= 1.0
        assert torch.all(torch.isfinite(Y))

        dtype = X.dtype
        device = X.device

        Y = Y.to(dtype=dtype, device=device)
        model = model.to(dtype=dtype, device=device)

        if acqf == "ts" and "n_candidates" not in acqf_kwargs:
            acqf_kwargs["n_candidates"] = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        tr_lb, tr_ub, x_center = self.get_trust_region_bounds(
            X,
            Y,
            weights=weights,
        )

        if acqf.lower() == "ts":
            n_candidates = acqf_kwargs.get("n_candidates", min(5000, max(2000, 200 * X.shape[-1])))
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            if inequality_constraints is not None:
                if bounds is None:
                    raise ValueError("`bounds` must be provided when using `inequality_constraints`.")

                constraint_mask = torch.ones(n_candidates, dtype=torch.bool, device=device)
                lower = bounds[0]
                upper = bounds[1]

                # Constraints are defined in physical parameter space, so filter there.
                X_cand_un = X_cand * (upper - lower) + lower

                for indices, coeffs, rhs in inequality_constraints:
                    lhs = X_cand_un[:, indices] @ coeffs # compute the left-hand side
                    per_constraint_mask = lhs <= rhs # compare left-hand side to right-hand side to get a mask
                    constraint_mask = constraint_mask & per_constraint_mask # AND across constraints

                X_cand_un = X_cand_un[constraint_mask] # keep only filtered candidates in unnormalized space

                # trigger a warning if the constraints are too strict, however, it's likely that there will be an error raised during the initial seeding
             
                if X_cand_un.shape[0] == 0:
                    raise RuntimeError(
                        "No candidates left after applying constraints. "
                        "Your trust region might be too small or your constraints too strict."
                    )

                X_cand = (X_cand_un - lower) / (upper - lower) 

            # Keep GP in raw target space and apply minimization only at acquisition time.
            posterior_transform = None
            if not self.maximize:
                posterior_transform = ScalarizedPosteriorTransform(
                    weights=torch.tensor([-1.0], dtype=dtype, device=device) # flip sign for minimization
                )

            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(
                model=model,
                posterior_transform=posterior_transform,
                replacement=False,
            )
            with torch.no_grad():  
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        elif acqf.lower() == "ei":
            if self.maximize:
                best_f = Y.max()
                ei = qLogExpectedImprovement(model, best_f, X_pending=X_pending)
            else:
                posterior_transform = ScalarizedPosteriorTransform(
                    weights=torch.tensor([-1.0], dtype=dtype, device=device) # flip sign for minimization
                )
                ei = qLogExpectedImprovement(
                    model=model,
                    best_f=Y.min(),
                    X_pending=X_pending,
                    posterior_transform=posterior_transform,
                )
            X_next, _ = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                # same BoTorch inequality tuples used in TS filtering
                inequality_constraints=inequality_constraints,
                **acqf_kwargs,
            )

        return X_next


class TuRBOGenerationNode(ExternalGenerationNode):
    """A generation node that uses the TuRBO algorithm to generate a set of
    candidate designs.

    Parameters
    ----------
    model_options : dict[str, Any]
        Configuration options passed to TuRBO.
    batch_size : int
        Number of candidates generated per TuRBO iteration.
    device : torch.device | None, optional
        Torch device used for tensors and GP fitting, by default None.
    dtype : torch.dtype | None, optional
        Torch dtype used for tensors and GP fitting, by default None.
    acqf : Literal["ts", "ei"], optional
        Acquisition function used for the optimization, by default ``"ts"``.
    acqf_kwargs : dict[str, Any] | None, optional
        Additional acquisition-function keyword arguments, by default None.
    name : str, optional
        Node name used in the Ax generation strategy, by default
        ``"TuRBOGenerationNode"``.
    maximize : bool, optional
        Whether the objective is maximized or minimized, by default True.
    """

    def __init__(
        self,
        model_options: dict[str, Any],
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        acqf: Literal["ts", "ei"] = "ts",
        acqf_kwargs: dict[str, Any] | None = None,
        name: str = "TuRBOGenerationNode",
        maximize: bool = True,
    ) -> None:
    
        if acqf_kwargs is None:
            acqf_kwargs = {}

        super().__init__(name=name)

        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.double

        self.model_options = model_options
        self.batch_size = batch_size
        self.acqf = acqf
        self.acqf_kwargs = acqf_kwargs
        self.state: TurboState | None = None
        self.X_turbo: torch.Tensor | None = None
        self.Y_turbo: torch.Tensor | None = None
        self.parameters: list[RangeParameter] | None = None
        self.bounds: torch.Tensor | None = None
        # Ax constraint objects from the experiment search space
        self.parameter_constraints: list[Any] | None = None
        # BoTorch tuples converted from Ax `self.parameter_constraints`
        self.inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]] | None = None
        self.maximize = maximize
        self.sobol = None
        # TuRBO batch points put into queue so Ax can consume them one by one
        self.candidate_queue: list[dict[str, float]] = []
        # Use rounded parameter tuples for duplicate checks against pending trials, precision for comparisoncan be configured via `param_key_precision` in `model_options` (default: 12, which is around the precision of a 64-bit float)
        self.param_key_precision = int(self.model_options.get("param_key_precision", 12))

    def _parse_inequality_constraints(
        self,
        parameter_names: list[str],
        parameter_constraints: list[Any] | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, float]] | None:
        """Convert Ax parameter constraints to BoTorch inequality tuples.

        Parameters
        ----------
        parameter_names : list[str]
            Ordered parameter names used in the search space.
        parameter_constraints : list[Any] | None
            Ax constraint objects from the search space.

        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor, float]] | None
            BoTorch inequality constraints of the form
            ``(indices, coeffs, rhs)`` or ``None`` if no constraints are
            available.

        Raises
        ------
        ValueError
            If a constraint references an unknown parameter name.
        """
        if not parameter_constraints:
            return None

        name_to_idx = {name: i for i, name in enumerate(parameter_names)}
        inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]] = []

        for constraint in parameter_constraints:
            constraint_dict = getattr(constraint, "constraint_dict", None)
            bound = getattr(constraint, "bound", None)
            if not constraint_dict or bound is None:
                continue

            indices = []
            coeffs = []
            for name, coeff in constraint_dict.items():
                if name not in name_to_idx:
                    raise ValueError(f"Unknown parameter '{name}' in constraint '{constraint}'.")
                indices.append(name_to_idx[name])
                coeffs.append(float(coeff))

            inequality_constraints.append(
                (
                    torch.tensor(indices, dtype=torch.long, device=self.device),
                    torch.tensor(coeffs, dtype=self.dtype, device=self.device),
                    float(bound),
                )
            )

        return inequality_constraints or None

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """Update the TuRBO state from completed Ax trials.

        Parameters
        ----------
        experiment : Experiment
            Ax experiment object containing the search space and trial objects.
        data : Data
            Ax data object containing completed trial results.

        Raises
        ------
        NotImplementedError
            If the search space contains parameters other than
            ``RangeParameter`` instances.
        ValueError
            If more than one objective metric is present.
        """
        search_space = experiment.search_space

        if any(not isinstance(p, RangeParameter) for p in search_space.parameters.values()):
            raise NotImplementedError("This method only supports RangeParameters in the search space.")

        parameter_names = list(search_space.parameters.keys())
        # metric_names = list(experiment.optimization_config.metrics.keys())  # pyright: ignore[reportOptionalMemberAccess]
        metric_names = experiment.optimization_config._objective.metric_names

        if self.parameters is None:
            self.parameters = list(search_space.parameters.values())  # pyright: ignore[reportAttributeAccessIssue]
            self.bounds = torch.tensor(
                [[p.lower for p in self.parameters], [p.upper for p in self.parameters]],
                dtype=self.dtype,
                device=self.device,
            )
            # would need a change if the constraints were not static, right now the are pulled and parsed once during node initialization
            self.parameter_constraints = getattr(search_space, "parameter_constraints", None) # pull the constraints from the Ax search space object
            self.inequality_constraints = self._parse_inequality_constraints( # convert Ax constraint objects to BoTorch tuples
                parameter_names=parameter_names,
                parameter_constraints=self.parameter_constraints,
            )

        if self.sobol is None:
            self.sobol = SobolEngine(len(parameter_names), scramble=True)

        # Initialize TuRBO state and data if it's the first call
        if self.state is None:
            self.state = TurboState(dim=len(parameter_names), batch_size=self.batch_size, maximize=self.maximize)

        if len(metric_names) != 1:
            raise ValueError("This generation node only supports a single metric.")

        # Try building tensors row-by-row from (sorted) completed trials so missing metrics or
        # invalid values are skipped without breaking X / Y alignment
        completed_trials = sorted(
            (trial for trial in experiment.trials.values() if trial.status == TrialStatus.COMPLETED),
            key=lambda trial: trial.index,
        )

        X_rows: list[torch.Tensor] = [] # parameter vectors
        Y_rows: list[torch.Tensor] = [] # metric values

        for trial in completed_trials:
            trial_parameters = trial.arm.parameters # dict for the specific trial  # pyright: ignore[reportAttributeAccessIssue]
            trial_df = data.df[data.df["trial_index"] == trial.index] # take only rows belonging to the specific trial
            filtered_df = trial_df[trial_df["metric_name"] == metric_names[0]] # keep only row for the objective metric(SingleObjective) 
            if filtered_df.empty:
                continue

            value = float(filtered_df["mean"].iloc[0])
            if not math.isfinite(value): # check for valid finite number (should catch NaN or inf vals)
                continue
                
            # conversion below after the filtering
            X_rows.append(
                torch.tensor(
                    [float(trial_parameters.get(name, 0.0)) for name in parameter_names],
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            Y_rows.append(
                torch.tensor(
                    [value],
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        if len(X_rows) == 0: # no valid completed trials
            return

        # Normalize X to [0, 1]^d
        X = torch.stack(X_rows)
        Y_new = torch.stack(Y_rows)
        X_normalized = self.to_unit_cube(X)
        self.state = self.state.update_state(Y_next=Y_new)

        self.candidate_queue.clear() 

        self.X_turbo = X_normalized
        self.Y_turbo = Y_new

    def to_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Convert a tensor of parameters to the unit cube

        Parameters
        ----------
        X : torch.Tensor
            Parameter tensor in physical search-space coordinates.

        Returns
        -------
        torch.Tensor
            Normalized tensor with values in the unit cube.

        Raises
        ------
        RuntimeError
            If the generator has not yet been initialized from the experiment
            search space.
        """
        if self.parameters is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        lower_bounds = torch.tensor(
            [p.lower for p in self.parameters],
            dtype=self.dtype,
            device=self.device,
        )

        upper_bounds = torch.tensor(
            [p.upper for p in self.parameters],
            dtype=self.dtype,
            device=self.device,
        )
        return (X - lower_bounds) / (upper_bounds - lower_bounds)

    def from_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Convert a tensor of parameters from the unit cube to the original space.

        Parameters
        ----------
        X : torch.Tensor
            Parameter tensor in normalized unit cube coordinates.

        Returns
        -------
        torch.Tensor
            Tensor mapped back to the original parameter bounds.

        Raises
        ------
        RuntimeError
            If the generator has not yet been initialized from the experiment
            search space.
        """
        if self.parameters is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        lower_bounds = torch.tensor(
            [p.lower for p in self.parameters],
            dtype=self.dtype,
            device=self.device,
        )
        upper_bounds = torch.tensor(
            [p.upper for p in self.parameters],
            dtype=self.dtype,
            device=self.device,
        )
        return X * (upper_bounds - lower_bounds) + lower_bounds

    def _key(self, params: dict[str, Any]) -> tuple[Any, ...]:
        """Create a stable key for duplicate checks against pending Ax trials, used in queue.

        Parameters
        ----------
        params : dict[str, Any]
            Candidate parameterization in physical coordinates.

        Returns
        -------
        tuple[Any, ...]
            Tuple representation used for duplicate detection.

        Raises
        ------
        RuntimeError
            If the generator has not yet been initialized from the experiment
            search space.
        """
        if self.parameters is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        key_vals: list[Any] = []
        for p in self.parameters:
            value = params[p.name]
            if "int" in str(p.parameter_type).lower():
                key_vals.append(int(value))
            else:
                key_vals.append(round(float(value), self.param_key_precision))
        return tuple(key_vals)

    def _enqueue_batch_from_turbo(self, X_pending: torch.Tensor | None) -> None:
        """Generate one TuRBO batch and enqueue candidates for Ax consumption.

        Parameters
        ----------
        X_pending : torch.Tensor | None
            Pending points in normalized space, or ``None`` if no trials are
            currently pending.

        Raises
        ------
        RuntimeError
            If the internal TuRBO state has not yet been initialized.
        """
        if self.X_turbo is None or self.Y_turbo is None or self.state is None or self.parameters is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        X_train, Y_train = self.X_turbo, self.Y_turbo
        
        # Fit a GP model in raw target space to preserve direct interpretability/export.
        # Fit the GP once for the full TuRBO batch.
        model = fit_gp(
            X_train,
            Y_train,
            normalize_inputs=False,  # Already normalized to [0,1]
            standardize_outputs=True,
            max_cholesky_size=self.model_options.get("max_cholesky_size", float("inf")),
        )

        # Generate candidates; TS/EI handle minimize via acquisition-time transforms.
        X_next = self.state.generate_batch(
            model=model,
            X=X_train,
            X_pending=X_pending,
            Y=Y_train,
            batch_size=self.batch_size,
            acqf=self.acqf,  # pyright: ignore[reportArgumentType]
            acqf_kwargs=self.acqf_kwargs,
            inequality_constraints=self.inequality_constraints,
            bounds=self.bounds,
        )

        X_next_unnormalized = self.from_unit_cube(X_next)
        # Convert the sample to a parameterization. Multiple candidate dicts can be queued.
        for candidate in X_next_unnormalized:
            params: dict[str, float] = {}
            for idx, p in enumerate(self.parameters):
                params[p.name] = float(candidate[idx].item())
            self.candidate_queue.append(params)

    @override
    def get_next_candidate(self, pending_parameters: list[TParameterization]) -> TParameterization:
        """Get the parameters for the next candidate configuration to evaluate.

        Parameters
        ----------
        pending_parameters : list[TParameterization]
            Parameters of candidates that have been suggested but not yet
            evaluated.

        Returns
        -------
        TParameterization
            Parameterization for the next candidate to evaluate.

        Raises
        ------
        RuntimeError
            If the internal TuRBO state has not yet been initialized.
        """
        if self.X_turbo is None or self.Y_turbo is None or self.state is None or self.parameters is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        if len(pending_parameters) > 0:
            X_pending = torch.zeros(len(pending_parameters), len(self.parameters), dtype=self.dtype, device=self.device)
            parameter_names = [p.name for p in self.parameters]
            for i, pending in enumerate(pending_parameters):
                X_pending[i, :] = torch.tensor(
                    [pending.get(name, 0.0) for name in parameter_names],
                    dtype=self.dtype,
                    device=self.device,
                )

            X_pending = self.to_unit_cube(X_pending)
        else:
            X_pending = None

        pending_keys: set[tuple[Any, ...]] = set()
        for pending in pending_parameters:
            try:
                pending_keys.add(self._key(dict(pending))) # convert into tuple
            except Exception:
                continue

        draws = 0
        max_draws = max(10, 5 * self.batch_size) # safety/avoid infinite loops when queued candidates are skipped because they duplicate pending one
        while True:
            if len(self.candidate_queue) == 0: # if queue is empty, generate a new batch
                self._enqueue_batch_from_turbo(X_pending=X_pending)

            params = self.candidate_queue.pop(0)
            key = self._key(params) 
            if key in pending_keys and draws < max_draws: 
                draws += 1
                continue
            return params


class TuRBOGlobalStoppingStrategy(BaseGlobalStoppingStrategy):
    """Global stopping strategy for TuRBO that checks if the trust region has converged."""

    def __init__(self, generation_node_name: str = "TuRBO", min_trials: int = 0, inactive_when_pending_trials: bool = True) -> None:
        """Initialize the TuRBO global stopping strategy.

        Parameters
        ----------
        generation_node_name : str, optional
            Name of the TuRBO generation node to monitor, by default
            ``"TuRBO"``.
        min_trials : int, optional
            Minimum number of completed trials before stopping is considered,
            by default 0.
        inactive_when_pending_trials : bool, optional
            Whether stopping checks are disabled while trials are pending,
            by default True.
        """
        self.generation_node_name = generation_node_name
        super().__init__(
            min_trials=min_trials,
            inactive_when_pending_trials=inactive_when_pending_trials,
        )

    def _should_stop_optimization(self, experiment: Experiment, client: Client) -> tuple[bool, str]:
        """Check if the TuRBO generation node has triggered a restart due to convergence (trust region below minimum).

        Parameters
        ----------
        experiment : Experiment
            Experiment object containing the state of the optimization.
        client : Client
            Client object used to interact with the optimization.

        Returns
        -------
        tuple[bool, str]
            tuple containing a boolean indicating whether to stop the optimization and a message explaining the reason.

        """
       
        # get the current generation node
        current_node = client._generation_strategy._curr  

        # check if the current node is the TuRBO node

        if isinstance(current_node, TuRBOGenerationNode) and current_node.state is not None:
            if current_node.state.restart_triggered:
                return True, "TuRBO has converged (trust region below minimum) with current best value {:.4f} and length scale {:.4e} <= {:.4e}.".format(current_node.state.best_value, current_node.state.length, current_node.state.length_min)
        return False, ""