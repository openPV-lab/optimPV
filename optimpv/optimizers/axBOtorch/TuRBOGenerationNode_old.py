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

import torch, gpytorch
from ax.core.data import Data
from ax.api.client import Client
from ax.core.experiment import Experiment
from ax.core.parameter import RangeParameter
from ax.core.trial_status import TrialStatus
from ax.core.types import TParameterization
from ax.exceptions.core import OptimizationShouldStop
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective, ScalarizedPosteriorTransform
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
    """Build and fit a GP model using BoTorch defaults.

    Args:
        X: Training inputs of shape (n, d).
        Y: Training targets of shape (n, 1).
        normalize_inputs: Whether to normalize inputs to [0, 1]^d.
        standardize_outputs: Whether to standardize outputs to zero mean, unit variance.
        max_cholesky_size: Max size for Cholesky decomposition.

    Returns:
        Fitted GP model.
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

class TurboConvergedError(OptimizationShouldStop):
    """Raised when TuRBO has converged (trust region below minimum)."""

    pass


class TurboState(BaseModel):
    """State of the TuRBO algorithm."""

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
        """Initialize optional fields based on the required fields."""
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
        """Get the trust region bounds for the TuRBO algorithm."""
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
        """Get the trust region data for the TuRBO algorithm."""
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
        """Generate a batch of candidates within the current trust region."""
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
                if X_cand_un.shape[0] < batch_size:
                    print(
                        "Reduced candidate size from "
                        f"{constraint_mask.shape[0]} to {X_cand_un.shape[0]} due to constraints. "
                        "This may lead to suboptimal results. Consider increasing n_candidates."
                    )
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
        """Initialize the generation node.

        Args:
            model_options: Options to pass to the GP model.
            batch_size: The batch size for generating new candidates.
            device: The device to use for the generation node.
            dtype: The dtype to use for the generation node.
            acqf: Acquisition function to use ("ts" for Thompson sampling,
                "ei" for Expected Improvement).
            acqf_kwargs: Keyword arguments for the acquisition function.
            name: The name of the generation node.
            maximize: Whether the generation node is maximizing or minimizing the objective
        """
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
        """Convert Ax constraint objects to BoTorch tuples (indices, coeffs, rhs)."""
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
        """Update the state of the generator with the experiment and data.

        Args:
            experiment: The experiment object.
            data: The data object.
        """
        search_space = experiment.search_space

        if any(not isinstance(p, RangeParameter) for p in search_space.parameters.values()):
            raise NotImplementedError("This method only supports RangeParameters in the search space.")

        parameter_names = list(search_space.parameters.keys())
        metric_names = list(experiment.optimization_config.metrics.keys())  # pyright: ignore[reportOptionalMemberAccess]

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

        # Get the data for the completed trials.
        num_completed_trials = len(experiment.trials_by_status[TrialStatus.COMPLETED])
        X = torch.zeros([num_completed_trials, len(parameter_names)], dtype=self.dtype, device=self.device)
        Y = torch.zeros([num_completed_trials, 1], dtype=self.dtype, device=self.device)

        for t_idx, trial in experiment.trials.items():
            if trial.status == TrialStatus.COMPLETED:
                trial_parameters = trial.arm.parameters  # pyright: ignore[reportAttributeAccessIssue]
                X[t_idx, :] = torch.tensor([trial_parameters.get(p, 0) for p in parameter_names], dtype=self.dtype)
                trial_df = data.df[data.df["trial_index"] == t_idx]
                filtered_df = trial_df[trial_df["metric_name"] == metric_names[0]]
                if not filtered_df.empty:
                    Y[t_idx, 0] = torch.tensor(filtered_df["mean"].item(), dtype=self.dtype)
                else:
                    Y[t_idx, 0] = torch.tensor(float("nan"), dtype=self.dtype)  # Handle missing data

        # Normalize X to [0, 1]^d
        X_normalized = self.to_unit_cube(X)
        Y_new = Y[~torch.isnan(Y).any(dim=1)]  # Filter out NaN values
        X_normalized = X_normalized[~torch.isnan(X_normalized).any(dim=1)]
        self.state = self.state.update_state(Y_next=Y_new)

        self.candidate_queue.clear() 

        self.X_turbo = X_normalized
        self.Y_turbo = Y_new

    def to_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Convert a tensor of parameters to the unit cube."""
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
        """Convert a tensor of parameters from the unit cube to the original space."""
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
        """Create a stable key for duplicate checks against pending Ax trials."""
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
        """Generate a TuRBO batch once and queue points for Ax ingestion."""
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

        Args:
            pending_parameters: A list of parameters of the candidates pending
                evaluation.

        Returns:
            A dictionary mapping parameter names to parameter values for the next
            candidate suggested by the method.
        """
        if self.X_turbo is None or self.Y_turbo is None or self.state is None or self.parameters is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        # use global stopping strategy to check if TuRBO has converged (trust region below minimum) and raise an error to stop the optimization if so, this will be caught in the optimizer and handled gracefully by stopping the optimization and logging a message to the user
        # if self.state.restart_triggered:
        #     raise TurboConvergedError(
        #         "TuRBO has converged (trust region below minimum). "
        #         "To continue optimization, create a new TurboGenerationNode instance with fresh initial points."
        #     )

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
        self.generation_node_name = generation_node_name
        super().__init__(
            min_trials=min_trials,
            inactive_when_pending_trials=inactive_when_pending_trials,
        )

    def _should_stop_optimization(self, experiment: Experiment, client: Client) -> tuple[bool, str]:
        """ Check if the TuRBO generation node has triggered a restart due to convergence (trust region below minimum).

        Parameters
        ----------
        experiment : Experiment
            experiment object containing the state of the optimization.
        client : Client
            client object used to interact with the optimization.

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