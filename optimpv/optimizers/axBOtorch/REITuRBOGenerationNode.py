"""Single-objective REI-TuRBO generation node for Ax."""

# The codes in this folder was adapted by @VMLC-PV and @FilipFekete from the original code, which can be found at

# https://github.com/Nobuo-Namura/regional-expected-improvement

# and the paper:

# Nobuo Namura and Sho Takemori, "Regional Expected Improvement for Efficient Trust Region Selection in High-Dimensional Bayesian Optimization," In Proceedings of the 39th AAAI Conference on Artificial Intelligence (2025).
# https://arxiv.org/abs/2412.11456

# @misc{namura2024regionalexpectedimprovementefficient,
#       title={Regional Expected Improvement for Efficient Trust Region Selection in High-Dimensional Bayesian Optimization}, 
#       author={Nobuo Namura and Sho Takemori},
#       year={2024},
#       eprint={2412.11456},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG},
#       url={https://arxiv.org/abs/2412.11456}, 
# }

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, override

import math
import gpytorch
import numpy as np
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.parameter import RangeParameter
from ax.core.trial_status import TrialStatus
from ax.core.types import TParameterization
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch_community.acquisition.rei import LogRegionalExpectedImprovement, qLogRegionalExpectedImprovement
from gpytorch.constraints import Interval
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch.quasirandom import SobolEngine


@dataclass
class TurboState:
    """Trust-region state used by the REI-TuRBO local TuRBO step.

    Attributes
    ----------
    dim : int
        Number of dimensions in the search space.
    batch_size : int
        Number of candidates generated per local batch.
    length : float
        Current trust-region side length in normalized space.
    length_min : float
        Minimum trust-region length before triggering a restart.
    length_max : float
        Maximum allowable trust-region length.
    failure_counter : int
        Number of consecutive non-improving updates.
    failure_tolerance : int
        Number of failures required to shrink the trust region.
    success_counter : int
        Number of consecutive improving updates.
    success_tolerance : int
        Number of successes required to expand the trust region.
    best_value : float
        Best objective value seen in the region so far.
    restart_triggered : bool
        Whether the trust region has converged and should be restarted.
    """

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 0
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        self.failure_tolerance = math.ceil( # set failure tolerance to ceil(4 / batch_size) as in the original REI-TuRBO implementation
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


class REITuRBO:
    """Node-oriented REI-TuRBO algorithm."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize the REI-TuRBO core logic.

        Parameters
        ----------
        device : torch.device
            Torch device used for tensors and model fitting.
        dtype : torch.dtype
            Torch dtype used for tensors and model fitting.
        """
        self.device = device
        self.dtype = dtype

    def initialize_state(self, dim: int, batch_size: int, Y_init: torch.Tensor) -> TurboState:
        """Create a fresh trust-region state from completed seed observations.

        Parameters
        ----------
        dim : int
            Number of dimensions in the search space.
        batch_size : int
            Number of candidates generated per local batch.
        Y_init : torch.Tensor
            Completed objective values collected during region seeding.

        Returns
        -------
        TurboState
            Initialized local trust-region state.
        """
        state = TurboState(dim=dim, batch_size=batch_size)
        if Y_init.numel() > 0:
            state.best_value = float(Y_init.max().item()) # take the best value from the seed data
        return state

    def update_state(self, state: TurboState, Y_next: torch.Tensor) -> TurboState:
        """Update local TuRBO trust-region state from newly completed points.

        Parameters
        ----------
        state : TurboState
            Current trust-region state.
        Y_next : torch.Tensor
            Newly observed objective values for the region.

        Returns
        -------
        TurboState
            Updated trust-region state.
        """
        if max(Y_next) > state.best_value:
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter == state.success_tolerance:
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:
            state.length /= 2.0
            state.failure_counter = 0

        state.best_value = max(state.best_value, max(Y_next).item())
        if state.length < state.length_min:
            state.restart_triggered = True
        return state

    def get_initial_points(self, dim: int, n_pts: int, seed: int | None = None) -> torch.Tensor:
        """Draw Sobol initial points in normalized space.

        Parameters
        ----------
        dim : int
            Number of search space dimensions.
        n_pts : int
            Number of Sobol points to draw.
        seed : int | None, optional
            Seed for the Sobol engine, by default None.

        Returns
        -------
        torch.Tensor
            Sobol points in normalized space.
        """
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        return sobol.draw(n=n_pts).to(dtype=self.dtype, device=self.device)

    # from the orignal implementation
    def select_sample(self, X: torch.Tensor, Y: torch.Tensor, n_gp_max: int = 2000) -> torch.Tensor:
        """Subsample global history before fitting the restart-seeding GP.

        Parameters
        ----------
        X : torch.Tensor
            Global history points in normalized space.
        Y : torch.Tensor
            Objective values corresponding to ``X``.
        n_gp_max : int, optional
            Maximum number of points kept for GP fitting, by default 2000.

        Returns
        -------
        torch.Tensor
            Boolean mask selecting the subset of history used for GP fitting.
        """
        if X.shape[0] <= n_gp_max:
            return torch.full([X.shape[0]], True, device=X.device, dtype=torch.bool)

        y_max = Y.max()
        y_min = Y.min()
        if torch.isclose(y_max, y_min):
            return torch.arange(X.shape[0], device=X.device) < n_gp_max

        regret = (y_max - Y) / (y_max - y_min)
        positive_regret = regret[regret > 0]
        if positive_regret.numel() == 0:
            return torch.arange(X.shape[0], device=X.device) < n_gp_max

        r_min2 = positive_regret.min()
        regret = (torch.log(regret + r_min2) - torch.log(r_min2)) / (
            torch.log(1 + r_min2) - torch.log(r_min2)
        )
        XY = torch.hstack([X, regret])
        mask = torch.full([XY.shape[0]], False, device=X.device, dtype=torch.bool)
        mask[regret[:, 0].argmin()] = True
        while int(mask.sum().item()) < n_gp_max:
            i_maxmin = torch.cdist(XY, XY[mask], p=2).min(axis=1)[0].argmax()
            mask[i_maxmin] = True
        return mask
    # from the orignal implemenatation
    def _standardize_y(self, Y: torch.Tensor) -> torch.Tensor:
        """Standardize observations using the original REI-TuRBO convention.

        Parameters
        ----------
        Y : torch.Tensor
            Objective values to standardize.

        Returns
        -------
        torch.Tensor
            Standardized objective values.
        """
        return (Y - Y.mean()) / Y.std()

    def generate_local_batch(
        self,
        state: TurboState,
        model: SingleTaskGP,
        X: torch.Tensor,
        Y: torch.Tensor,
        batch_size: int,
        *,
        n_candidates: int | None = None,
        num_restarts: int = 10,
        raw_samples: int = 512,
        acqf: Literal["TS", "EI"] = "TS",
        inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]] | None = None,
        bounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a local TuRBO batch inside the active trust region.

        Parameters
        ----------
        state : TurboState
            Local trust-region state.
        model : SingleTaskGP
            Fitted local Gaussian process model.
        X : torch.Tensor
            Completed points for the region in normalized space.
        Y : torch.Tensor
            Objective values corresponding to ``X``.
        batch_size : int
            Number of candidates to generate for each iteration.
        n_candidates : int | None, optional
            Number of Thompson candidates to draw before selection, by default
            None.
        num_restarts : int, optional
            Number of multistart restarts for EI optimization, by default 10.
        raw_samples : int, optional
            Number of raw samples for EI optimization, by default 512.
        acqf : Literal["TS", "EI"], optional
            Local acquisition function, by default ``"TS"``.
        inequality_constraints : list[tuple[torch.Tensor, torch.Tensor, float]] | None, optional
            BoTorch style inequality constraints, by default None.
        bounds : torch.Tensor | None, optional
            Physical space bounds used when filtering constrained TS
            candidates, by default None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Candidate points in normalized space with their acquisition values.

        Raises
        ------
        ValueError
            If constraints are supplied without bounds.
        RuntimeError
            If all Thompson candidates are filtered out by constraints.
        """
        assert acqf in ("TS", "EI")
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "TS":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            mask = torch.rand(n_candidates, dim, dtype=self.dtype, device=self.device) <= min(20.0 / dim, 1.0)
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            if inequality_constraints is not None:
                if bounds is None:
                    raise ValueError("`bounds` must be provided when using `inequality_constraints`.")
                lower = bounds[0]
                upper = bounds[1]
                constraint_mask = torch.ones(n_candidates, dtype=torch.bool, device=self.device)
                X_cand_un = X_cand * (upper - lower) + lower
                for indices, coeffs, rhs in inequality_constraints:
                    lhs = X_cand_un[:, indices] @ coeffs
                    constraint_mask = constraint_mask & (lhs <= rhs)
                X_cand = X_cand[constraint_mask]
                if X_cand.shape[0] == 0:
                    raise RuntimeError("No TS candidates left after applying constraints.")

            thompson_sampling = MaxPosteriorSampling(model=model, replacement=True)
            with torch.no_grad():
                posterior = thompson_sampling.model.posterior(
                    X_cand,
                    observation_noise=None,
                    posterior_transform=None,
                )
                samples = posterior.rsample(sample_shape=torch.Size([batch_size]))
                X_next = thompson_sampling.maximize_samples(X_cand, samples, batch_size)
                acq_value = torch.max(samples, dim=1)[0].reshape((-1, 1))
            return X_next, acq_value

        elif acqf == "EI":
            ei = qLogExpectedImprovement(model, Y.max())
            return optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                inequality_constraints=inequality_constraints,
            )

    def fit_local_model(
        self,
        X_region: torch.Tensor,
        Y_region: torch.Tensor,
        *,
        max_cholesky_size: int,
        noise_constraint: tuple[float, float],
    ) -> tuple[SingleTaskGP, torch.Tensor]:
        """Fit the local TuRBO GP.

        Parameters
        ----------
        X_region : torch.Tensor
            Completed points for one trust region in normalized space.
        Y_region : torch.Tensor
            Objective values corresponding to ``X_region``.
        max_cholesky_size : int
            Maximum matrix size for Cholesky decomposition in GPyTorch.
        noise_constraint : tuple[float, float]
            Lower and upper bound for the noise level.

        Returns
        -------
        tuple[SingleTaskGP, torch.Tensor]
            Fitted GP model and standardized training targets.
        """
        train_Y = self._standardize_y(Y_region)
        likelihood = GaussianLikelihood(
            noise_constraint=Interval(noise_constraint[0], noise_constraint[1])
        )
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=X_region.shape[-1],
                lengthscale_constraint=Interval(0.005, 4.0),
            )
        )
        model = SingleTaskGP(
            X_region,
            train_Y,
            covar_module=covar_module,
            likelihood=likelihood,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            fit_gpytorch_mll(mll)
        return model, train_Y

    # main REI/qREI seeding
    def sampling_from_global_model(
        self,
        X_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        *,
        bounds: torch.Tensor,
        n_init: int,
        q_batch: int = 1,
        rng: np.random.Generator,
        max_cholesky_size: int,
        length_init: float = 0.8,
        min_inferred_noise_level: float = 1e-4,
        racqf: Literal["REI", "QREI", "EI"] = "REI",
        rei_n_region: int = 128,
        rei_raw_samples: int = 512,
        rei_num_restarts: int = 10,
        rei_mc_samples: int = 128,
    ) -> torch.Tensor:
        """Select fresh trust-region seed batches from the global history model.

        Parameters
        ----------
        X_hist : torch.Tensor
            Global history points in normalized space.
        Y_hist : torch.Tensor
            Objective values corresponding to ``X_hist``.
        bounds : torch.Tensor
            Normalized search-space bounds.
        n_init : int
            Number of seed points proposed per region.
        q_batch : int, optional
            Number of regions seeded jointly, by default 1.
        rng : np.random.Generator
            NumPy random generator used for Sobol seeding.
        max_cholesky_size : int
            Maximum matrix size for Cholesky decomposition in GPyTorch.
        length_init : float, optional
            Initial trust-region length,
            by default 0.8.
        min_inferred_noise_level : float, optional
            Lower bound on inferred GP noise, by default 1e-4.
        racqf : Literal["REI", "QREI", "EI"], optional
            Regional acquisition function, by default ``"REI"``.
        rei_n_region : int, optional
            Number of regional design points used by REI, by default 128.
        rei_raw_samples : int, optional
            Number of raw samples for acquisition optimization, by default 512.
        rei_num_restarts : int, optional
            Number of restarts for acquisition optimization, by default 10.
        rei_mc_samples : int, optional
            Number of MC samples used by qREI, by default 128.

        Returns
        -------
        torch.Tensor
            Seed points in normalized space for one or more regions.

        Raises
        ------
        ValueError
            If an unsupported regional acquisition function is requested.
        """
        train_Y = self._standardize_y(Y_hist)

        covar_module = MaternKernel(
            nu=2.5,
            ard_num_dims=X_hist.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
        covar_module = ScaleKernel(covar_module, outputscale_prior=GammaPrior(2.0, 0.15))
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(
                min_inferred_noise_level,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )
        model = SingleTaskGP(
            X_hist,
            train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        seed = int(rng.integers(low=0, high=2**16, dtype=np.int64))
        sobol = SobolEngine(dimension=X_hist.shape[-1], scramble=True, seed=seed)
        X_dev = sobol.draw(n=rei_n_region).to(dtype=self.dtype, device=self.device)
        X_dev[torch.argmin(torch.sum((X_dev - 0.5) ** 2, axis=1)), :] = 0.5

        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            fit_gpytorch_mll(mll)

            # Build the regional acquisition function
            if racqf == "QREI":
                racq_function = qLogRegionalExpectedImprovement(
                    X_dev=X_dev,
                    model=model,
                    best_f=train_Y.max(),
                    sampler=SobolQMCNormalSampler(sample_shape=torch.Size([rei_mc_samples])),
                    length=length_init,
                    bounds=bounds,
                )
            elif racqf == "REI":
                racq_function = LogRegionalExpectedImprovement(
                    X_dev=X_dev,
                    model=model,
                    best_f=train_Y.max(),
                    length=length_init,
                    bounds=bounds,
                )
            elif racqf == "EI":
                racq_function = LogExpectedImprovement(model, train_Y.max())
            else:
                raise ValueError(f"Unsupported regional acquisition function '{racqf}'.")

            # Optimize the acquisition function to find candidate center(s)
            candidates, _ = optimize_acqf(
                acq_function=racq_function,
                bounds=bounds,
                q=q_batch,
                num_restarts=rei_num_restarts,
                raw_samples=rei_raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )
            X_center = candidates.detach()

        if n_init <= 1:
            return X_center.clone()

        seed = int(rng.integers(low=0, high=2**16, dtype=np.int64))
        sobol = SobolEngine(dimension=X_hist.shape[-1], scramble=True, seed=seed)
        X_seed = torch.empty([0, X_hist.shape[-1]], dtype=self.dtype, device=self.device)
        X_min = (X_center - 0.5 * length_init).clamp_min(bounds[0])
        X_max = (X_center + 0.5 * length_init).clamp_max(bounds[1])
        for center_idx in range(q_batch):
            X_init = sobol.draw(n_init - 1).to(dtype=self.dtype, device=self.device)
            X_seed = torch.cat(
                (
                    X_seed,
                    X_center[center_idx : center_idx + 1],
                    X_init * (X_max[center_idx : center_idx + 1] - X_min[center_idx : center_idx + 1])
                    + X_min[center_idx : center_idx + 1],
                ),
                dim=0,
            )
        return X_seed


@dataclass
class REITurboNodeState:
    """Runtime state owned by the REI-TuRBO node.

    Attributes
    ----------
    turbo_states : dict[int, TurboState]
        Local TuRBO state for each initialized trust region.
    seen_trial_indices : set[int]
        Completed Ax trial indices already ingested by the node.
    pending_region_index_by_key : dict[tuple[Any, ...], int]
        Mapping from candidate keys to the region that proposed them while
        those candidates are still pending in Ax.
    candidate_queue : list[dict[str, Any]]
        Queue of candidates ready to be returned to Ax.
    active_region_indices : list[int]
        Region indices currently active in the optimizer.
    next_region_index : int
        Next unique region index assigned when creating a new trust region.
    """

    turbo_states: dict[int, TurboState] = field(default_factory=dict) # state of each active trust region
    seen_trial_indices: set[int] = field(default_factory=set) # indices of trials that have been evaluated
    pending_region_index_by_key: dict[tuple[Any, ...], int] = field(default_factory=dict) # maps pending candidates to their proposed trust region
    candidate_queue: list[dict[str, Any]] = field(default_factory=list) # queue of candidates ready to be proposed to Ax
    active_region_indices: list[int] = field(default_factory=list) 
    next_region_index: int = 0


class REITuRBOGenerationNode(ExternalGenerationNode):
    """ExternalGenerationNode that uses REI to seed a TuRBO trust region.

    Parameters
    ----------
    model_options : dict[str, Any] | None, optional
        Options controlling REI seeding, local TuRBO generation, and numeric
        settings, by default None.
    batch_size : int, optional
        Number of candidates proposed per generation step, by default 1.
    device : torch.device | None, optional
        Torch device used for tensors and model fitting, by default None.
    dtype : torch.dtype | None, optional
        Torch dtype used for tensors and model fitting, by default None.
    acqf : Literal["ts", "ei"], optional
        Acquisition function used for local TuRBO generation, by default ``"ts"``.
    name : str, optional
        Name used by the Ax generation strategy, by default ``"REITuRBOGenerationNode"``.
    maximize : bool, optional
        Whether the objective is maximized or minimized, by default True.
    """

    def __init__(
        self,
        model_options: Optional[dict[str, Any]] = None,
        batch_size: int = 1,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        acqf: Literal["ts", "ei"] = "ts", # acqf for local TuRBO generation
        name: str = "REITuRBOGenerationNode",
        maximize: bool = True,
    ) -> None:
           
        super().__init__(name=name)

        self.model_options = dict(model_options or {})
        if device is None and "torch_device" in self.model_options:
            device = self.model_options["torch_device"]
        if dtype is None and "torch_dtype" in self.model_options:
            dtype = self.model_options["torch_dtype"]
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.double

        self.batch_size = int(batch_size)
        self.local_acqf = acqf.upper()
        self.maximize = bool(maximize)
        self.n_trust_regions = int(self.model_options.get("n_trust_regions", 1)) # Turbo-m if >1, Turbo-1 if =1
        # `region_init_points` is REI-TuRBO-specific and refers to the number of proposed points per trust region. 
        self.region_init_points = int(self.model_options.get("region_init_points", max(4, self.batch_size)))
        self.n_gp_max = int(self.model_options.get("n_gp_max", 2000))
        self.max_cholesky_size = int(self.model_options.get("max_cholesky_size", 2000))
        self.noise_constraint = tuple(self.model_options.get("noise_constraint", (1e-4, 1e0)))
        self.local_num_restarts = int(self.model_options.get("num_restarts", 10))
        self.local_raw_samples = int(self.model_options.get("raw_samples", 512))
        self.n_candidates = self.model_options.get("n_candidates")
        self.racqf = str(self.model_options.get("racqf", "REI")).upper() # acqf for seeding trust regions REI if Turbo-1, racqf if Turbo-m 
        self.rei_n_region = int(self.model_options.get("rei_n_region", 128))
        self.rei_num_restarts = int(self.model_options.get("rei_num_restarts", 10))
        self.rei_raw_samples = int(self.model_options.get("rei_raw_samples", 512))
        self.rei_mc_samples = int(self.model_options.get("rei_mc_samples", 128))
        self.min_inferred_noise_level = float(self.model_options.get("min_inferred_noise_level", 1e-4))
        self.param_key_precision = int(self.model_options.get("param_key_precision", 12))
        self.seed = int(self.model_options.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        self.engine = REITuRBO(device=self.device, dtype=self.dtype)
        self.state = REITurboNodeState()

        self.parameters: list[RangeParameter] | None = None
        self.param_names: list[str] = []
        self.int_param_names: set[str] = set()
        self.bounds_tensor: torch.Tensor | None = None
        self.normalized_bounds: torch.Tensor | None = None
        self.parameter_constraints: list[Any] | None = None
        self.inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]] | None = None

        self.X_hist: torch.Tensor | None = None
        self.Y_hist: torch.Tensor | None = None
        self.X_turbo_by_region: dict[int, torch.Tensor] = {}
        self.Y_turbo_by_region: dict[int, torch.Tensor] = {}

    # Ax adapter logic
    def _initialize_ax_parameter_cache(self, experiment: Experiment) -> None:
        """Pull parameter metadata from Ax search space. Cache the metadata in the same order used for tensors.
        It caches parameter order, integer parameters, lower/upper bounds, normalized bounds, and parameter constraints.

        Parameters
        ----------
        experiment : Experiment
            Ax experiment object where search space defines the parameter metadata.

        Raises
        ------
        NotImplementedError
            If the search space contains parameters other than ``RangeParameter`` instances.
        """
        search_space = experiment.search_space
        if any(not isinstance(p, RangeParameter) for p in search_space.parameters.values()):
            raise NotImplementedError("REITuRBOGenerationNode supports only RangeParameters.")

        self.parameters = list(search_space.parameters.values())  # type: ignore[arg-type]
        self.param_names = [p.name for p in self.parameters]
        self.int_param_names = {
            p.name for p in self.parameters if "int" in str(p.parameter_type).lower()
        }
        lb = [float(p.lower) for p in self.parameters]
        ub = [float(p.upper) for p in self.parameters]
        self.bounds_tensor = torch.tensor([lb, ub], dtype=self.dtype, device=self.device)
        self.normalized_bounds = torch.stack(
            [
                torch.zeros(len(self.parameters), dtype=self.dtype, device=self.device),
                torch.ones(len(self.parameters), dtype=self.dtype, device=self.device),
            ]
        )
        self.parameter_constraints = getattr(search_space, "parameter_constraints", None)
        self.inequality_constraints = self._parse_inequality_constraints(
            parameter_names=self.param_names,
            parameter_constraints=self.parameter_constraints,
        )

    # Ax adapter logic
    def _parse_inequality_constraints(
        self,
        parameter_names: list[str],
        parameter_constraints: list[Any] | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, float]] | None:
        """Convert Ax parameter constraints to BoTorch inequality tuples.

        Parameters
        ----------
        parameter_names : list[str]
            Ordered parameter names used by the node tensors.
        parameter_constraints : list[Any] | None
            Ax parameter-constraint objects from the search space.

        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor, float]] | None
            BoTorch inequality constraints in ``(indices, coeffs, rhs)``
            format, or None if no valid constraints are available.

        Raises
        ------
        ValueError
            If a constraint references an unknown parameter name.
        """
        if not parameter_constraints:
            return None

        name_to_idx = {name: idx for idx, name in enumerate(parameter_names)}
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

    # Ax adapter logic
    def _key(self, params: dict[str, Any]) -> tuple[Any, ...]:
        """Create a stable key for duplicate checks against pending Ax trials, used in queue."""

        key_vals: list[Any] = []
        for name in self.param_names:
            value = params[name]
            if name in self.int_param_names:
                key_vals.append(int(value))
            else:
                key_vals.append(round(float(value), self.param_key_precision))
        return tuple(key_vals)

    def _pending_count_for_region(self, region_index: int) -> int:
        """Count pending Ax candidates currently associated with one region."""
        return sum(1 for idx in self.state.pending_region_index_by_key.values() if idx == region_index)

    def _seed_acqf_for_batch(self, q_batch: int) -> str:
        """Decision branching point. Use qREI only for joint multi-region seeding and fall back to REI for single-region seeding."""
        if q_batch > 1:
            return "QREI"
        if self.racqf == "QREI":
            return "REI"
        return self.racqf

    def _is_region_initialized(self, region_index: int) -> bool:
        """Check if a trust region already has a local TuRBO state."""
        return region_index in self.state.turbo_states

    def _active_initialized_regions(self) -> list[int]:
        """Return active trust regions that have local TuRBO state."""
        return [region_index for region_index in self.state.active_region_indices if self._is_region_initialized(region_index)]

    def _active_uninitialized_regions(self) -> list[int]:
        """Return active trust regions that are still collecting seed points."""
        return [region_index for region_index in self.state.active_region_indices if not self._is_region_initialized(region_index)]
    # global
    def _append_history(self, X_new: torch.Tensor, Y_new: torch.Tensor) -> None:
        """Append new completed trials to the global history used for REI restart seeding.

        Parameters
        ----------
        X_new : torch.Tensor
            Newly completed points in normalized space.
        Y_new : torch.Tensor
            Objective values corresponding to ``X_new``.
        """
        if self.X_hist is None or self.Y_hist is None:
            self.X_hist = X_new
            self.Y_hist = Y_new
        else:
            self.X_hist = torch.cat((self.X_hist, X_new), dim=0)
            self.Y_hist = torch.cat((self.Y_hist, Y_new), dim=0)
    # local specific to each trust region
    def _append_region_data(self, region_index: int, X_new: torch.Tensor, Y_new: torch.Tensor) -> None:
        """Append newly completed trials belonging to one active trust region.

        Parameters
        ----------
        region_index : int
            Trust region id.
        X_new : torch.Tensor
            Newly completed points for the region in normalized space.
        Y_new : torch.Tensor
            Objective values corresponding to ``X_new``.
        """
        if region_index not in self.X_turbo_by_region or region_index not in self.Y_turbo_by_region:
            self.X_turbo_by_region[region_index] = X_new
            self.Y_turbo_by_region[region_index] = Y_new
        else:
            self.X_turbo_by_region[region_index] = torch.cat((self.X_turbo_by_region[region_index], X_new), dim=0)
            self.Y_turbo_by_region[region_index] = torch.cat((self.Y_turbo_by_region[region_index], Y_new), dim=0)
    # corresponds to orginal turbo-m behaviour where restarted region is replaced by a fresh ona and (tr_active is updated)
    def _remove_region(self, region_index: int) -> None:
        """Drop a finished trust region before replacing it with a fresh seeded region.

        Parameters
        ----------
        region_index : int
            Trust region id to remove.
        """
        if region_index in self.state.active_region_indices:
            self.state.active_region_indices = [idx for idx in self.state.active_region_indices if idx != region_index]
        self.state.turbo_states.pop(region_index, None)
        self.X_turbo_by_region.pop(region_index, None)
        self.Y_turbo_by_region.pop(region_index, None)
    # entry point for seeding new trust regions 
    def _enqueue_region_seed_batch(self, region_indices: list[int]) -> None:
        """Use the offloaded REI global model to seed one or more trust regions. 
        And enqueue the proposed candidates into the candidate queue.
        Adapted verion of sampling seed points from the global REI/qREI model and assigning them to new trust regions.

        Parameters
        ----------
        region_indices : list[int]
            Region iDs to seed from the global model.

        Raises
        ------
        RuntimeError
            If global history or normalized bounds are not initialized.
        """
        if self.normalized_bounds is None or self.X_hist is None or self.Y_hist is None: # check global history exists
            raise RuntimeError("REI-TuRBO requires completed trials before seeding a trust region.")

        if len(region_indices) == 0:
            return

        # check proposed regions are empty, how many points they have, and how many pending trials they have
        for region_index in region_indices:
            current_completed = int(self.X_turbo_by_region[region_index].shape[0]) if region_index in self.X_turbo_by_region else 0
            if current_completed >= self.region_init_points: # if the region has enough completed points skip it
                continue

        # Offload the regional restart/initialization logic to the dedicated engine.
        mask = self.engine.select_sample(self.X_hist, self.Y_hist, self.n_gp_max) # select a subset of the global history if too big (from original implemntation)
        X_seed = self.engine.sampling_from_global_model(
            self.X_hist[mask],
            self.Y_hist[mask],
            bounds=self.normalized_bounds,
            n_init=self.region_init_points,
            q_batch=len(region_indices), # number of regions initialized
            rng=self.rng,
            max_cholesky_size=self.max_cholesky_size,
            length_init=TurboState(dim=len(self.param_names), batch_size=self.batch_size).length,
            min_inferred_noise_level=self.min_inferred_noise_level,
            racqf=self._seed_acqf_for_batch(len(region_indices)), # TuRBO-m seeding with q>1 gets QREI, TuRBO-1 seeding with q=1 gets racqf (default REI)
            rei_n_region=self.rei_n_region,
            rei_raw_samples=self.rei_raw_samples,
            rei_num_restarts=self.rei_num_restarts,
            rei_mc_samples=self.rei_mc_samples,
        )
        # X_seed comes back in the normalized space
        X_seed_un = self.from_unit_cube(X_seed) # convert back to the Ax space
        for offset, candidate in enumerate(X_seed_un): # Iterate through all generated seed points 
            region_index = region_indices[offset // self.region_init_points] # assign each block of `region_init_points` candidates to one region
            params: dict[str, Any] = {}
            for idx, name in enumerate(self.param_names):
                raw_value = float(candidate[idx].item())
                params[name] = int(round(raw_value)) if name in self.int_param_names else raw_value
            self.state.candidate_queue.append( # enqueue the initial candidates (equivalent of sampling_from_global_model(...) in the original code)
                {
                    "params": params,
                    "region_index": region_index,
                }
            )
    # follows the original TuRBO-m local generation logic
    def _enqueue_local_batch(self, X_pending: torch.Tensor | None) -> None:
        """Use the offloaded local TuRBO engine across all active trust regions.
        Implements multi-region local candidate generation. 
        Mainly fits a local GP, generates local TuRBO batch, attaches acq scores. 
        Pools either pointwise candidates or region batches depending on the returned score values. 

        Parameters
        ----------
        X_pending : torch.Tensor | None
            Pending points in normalized space, or None if no trials are currently pending.

        Raises
        ------
        RuntimeError
            If bounds are not initialized or no active initialized region can produce a candidate batch.
        """
        if self.bounds_tensor is None:
            raise RuntimeError("Bounds not initialized.")

        pointwise_candidates: list[dict[str, Any]] = [] # list of candidate dicts collected across all regions, each candidate scored individually for TS/single-point EI
        region_candidates: list[dict[str, Any]] = [] # list of region batch scores collected across all regions, each batch gets one score for qEI/qLogEI with q>1
        # loop through active regions
        for region_index in self._active_initialized_regions():
            turbo_state = self.state.turbo_states[region_index]
            X_region = self.X_turbo_by_region.get(region_index)
            Y_region = self.Y_turbo_by_region.get(region_index)
            if X_region is None or Y_region is None:
                continue
            model, train_Y = self.engine.fit_local_model(
                X_region,
                Y_region,
                max_cholesky_size=self.max_cholesky_size,
                noise_constraint=self.noise_constraint,  # type: ignore[arg-type]
            )
            X_next, acq_value = self.engine.generate_local_batch(
                state=turbo_state,
                model=model,
                X=X_region,
                Y=train_Y,
                batch_size=self.batch_size,
                n_candidates=self.n_candidates,
                num_restarts=self.local_num_restarts,
                raw_samples=self.local_raw_samples,
                acqf=self.local_acqf,
                inequality_constraints=self.inequality_constraints,
                bounds=self.bounds_tensor,
            )
            X_next_un = self.from_unit_cube(X_next)
            acq_flat = acq_value.reshape(-1) # 1D tensor from the acq function output
            params_list: list[dict[str, Any]] = []
            for candidate in X_next_un:
                params: dict[str, Any] = {}
                for idx, name in enumerate(self.param_names):
                    raw_value = float(candidate[idx].item())
                    params[name] = int(round(raw_value)) if name in self.int_param_names else raw_value
                params_list.append(params)

            # TS returns one score per selected point, 
            # so we implement global pooling across all candidate rows.
            # qEI/qLogEI with q>1 returns one joint score for the whole region batch,
            # so rank region batches against each other and enqueue the winning batch.
            if acq_flat.numel() == len(params_list):  # Pointwise scores: TS and EI with q=1
                for row_idx, params in enumerate(params_list):
                    pointwise_candidates.append(
                        {
                            "params": params,
                            "region_index": region_index,
                            "score": float(acq_flat[row_idx].item()),
                        }
                    )
            elif acq_flat.numel() == 1:  # Batch score: qEI/qLogEI with q>1
                region_candidates.append(
                    {
                        "params_list": params_list,
                        "region_index": region_index,
                        "score": float(acq_flat[0].item()),
                    }
                )
            else:
                raise RuntimeError(
                    "Unexpected acquisition score shape returned by REI-TuRBO local generation."
                )

        if len(pointwise_candidates) == 0 and len(region_candidates) == 0:
            raise RuntimeError("No initialized REI-TuRBO regions are ready to generate local candidates.")

        # Pointwise scoring path used by TS and single-point EI.
        if len(pointwise_candidates) > 0:
            pointwise_candidates.sort(key=lambda item: item["score"], reverse=True)
            for item in pointwise_candidates[: self.batch_size]:
                self.state.candidate_queue.append(
                    {
                        "params": item["params"],
                        "region_index": item["region_index"],
                    }
                )
            return

        # Batchwise scoring path used by batched qEI/qLogEI.
        # if the batch_size = x there will be x candidates per region, but only the x number of points from the best region will be proposed to Ax, meaning points only from one region are proposed
        region_candidates.sort(key=lambda item: item["score"], reverse=True) # sort regions by score and enqueue candidates from the best regions
        remaining_points = self.batch_size
        for item in region_candidates:
            if remaining_points <= 0:
                break
            for params in item["params_list"][:remaining_points]:
                self.state.candidate_queue.append(
                    {
                        "params": params,
                        "region_index": item["region_index"],
                    }
                )
            remaining_points -= len(item["params_list"][:remaining_points])

    # logic similar to orignal turbo-m restart where restarted regions are replaced by fresh seeded regions from the global model.
    # exception is that with the Ax adapter, the restart trigger is only checked when the candidate queue is empty 
    # and new candidates need to be generated. 
    def _restart_regions(self, region_indices: list[int]) -> None:
        """Replace restarted trust regions with freshly seeded regions from the global history with new iDs."""
        
        if len(region_indices) == 0:
            return

        for region_index in region_indices:
            self._remove_region(region_index)

        new_region_indices: list[int] = []
        for _ in region_indices:
            new_region_indices.append(self.state.next_region_index)
            self.state.active_region_indices.append(self.state.next_region_index)
            self.state.next_region_index += 1

        self._enqueue_region_seed_batch(new_region_indices)

    def to_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Map points from physical parameter space to normalized space.

        Parameters
        ----------
        X : torch.Tensor
            Points in physical parameter coordinates.

        Returns
        -------
        torch.Tensor
            Points normalized to the unit cube.

        Raises
        ------
        RuntimeError
            If bounds have not been initialized.
        """
        if self.bounds_tensor is None:
            raise RuntimeError("Bounds not initialized.")
        return (X - self.bounds_tensor[0]) / (self.bounds_tensor[1] - self.bounds_tensor[0])

    def from_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Map points from normalized space back to physical parameter space.

        Parameters
        ----------
        X : torch.Tensor
            Points in normalized unit-cube coordinates.

        Returns
        -------
        torch.Tensor
            Points in physical parameter coordinates.

        Raises
        ------
        RuntimeError
            If bounds have not been initialized.
        """
        if self.bounds_tensor is None:
            raise RuntimeError("Bounds not initialized.")
        return X * (self.bounds_tensor[1] - self.bounds_tensor[0]) + self.bounds_tensor[0]

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """Take newly completed Ax trials and update global/local REI-TuRBO state.

        Parameters
        ----------
        experiment : Experiment
            Ax experiment object containing the search space and trial objects.
        data : Data
            Ax data object containing completed trial results.

        Raises
        ------
        ValueError
            If the experiment has no optimization config or has more than one objective metric.
        """
        if self.parameters is None: # initialize on first call
            self._initialize_ax_parameter_cache(experiment=experiment)

        if experiment.optimization_config is None:
            raise ValueError("Experiment has no optimization_config configured.")
        # metric_names = list(experiment.optimization_config.metrics.keys())
        metric_names = experiment.optimization_config._objective.metric_names
        if len(metric_names) != 1:
            raise ValueError("REITuRBOGenerationNode only supports a single metric.")

        completed_trials = sorted(
            (trial for trial in experiment.trials.values() if trial.status == TrialStatus.COMPLETED),
            key=lambda trial: trial.index,
        )
        if len(completed_trials) == 0:
            return

        X_hist_rows: list[torch.Tensor] = []
        Y_hist_rows: list[torch.Tensor] = []
        region_updates: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {} # new data grouped by region id
        new_trial_indices: list[int] = []

        for trial in completed_trials:
            if trial.index in self.state.seen_trial_indices:
                continue

            params = trial.arm.parameters  # type: ignore[attr-defined]
            trial_df = data.df[data.df["trial_index"] == trial.index]
            row = trial_df[trial_df["metric_name"] == metric_names[0]]
            if row.empty:
                continue

            x_raw = torch.tensor(
                [float(params[name]) for name in self.param_names],
                dtype=self.dtype,
                device=self.device,
            ).unsqueeze(0)
            X_point = self.to_unit_cube(x_raw).squeeze(0)
            value = float(row["mean"].iloc[0])
            Y_point = torch.tensor(
                [[value if self.maximize else -value]], # flip metric sign for minimization
                dtype=self.dtype,
                device=self.device,
            )

            X_hist_rows.append(X_point.unsqueeze(0))
            Y_hist_rows.append(Y_point)
            new_trial_indices.append(trial.index)

            region_index = self.state.pending_region_index_by_key.pop(self._key(params), None) # points back to the region that proposed them
            # Ax only stores the trial parameters and results, it does not have a notion of trust regions
            if region_index is not None and region_index in self.state.active_region_indices: # if the completed trial belongs to an active region, group it for the local model update
                if region_index not in region_updates:
                    region_updates[region_index] = ([], [])
                region_updates[region_index][0].append(X_point.unsqueeze(0))
                region_updates[region_index][1].append(Y_point)

        if len(X_hist_rows) == 0:
            return

        # append new valid completed trials to global history 
        X_hist_new = torch.cat(X_hist_rows, dim=0)
        Y_hist_new = torch.cat(Y_hist_rows, dim=0)
        self._append_history(X_hist_new, Y_hist_new)

        # Keep queued REI seed points while regions are still being initialized.
        # Once a local TuRBO state exists, newly completed data should invalidate
        # any queued local batch so the next batch is generated from the updated GP.
        if any(self._is_region_initialized(region_index) for region_index in region_updates):
            self.state.candidate_queue = [
                item for item in self.state.candidate_queue if not self._is_region_initialized(int(item["region_index"]))
            ]

        # each region that got new data append the new data to the local history and updates the local TuRBO state
        for region_index, (X_rows, Y_rows) in region_updates.items():
            X_region_new = torch.cat(X_rows, dim=0)
            Y_region_new = torch.cat(Y_rows, dim=0)
            self._append_region_data(region_index, X_region_new, Y_region_new)

            X_region = self.X_turbo_by_region.get(region_index)
            Y_region = self.Y_turbo_by_region.get(region_index)
            if X_region is None or Y_region is None:
                continue
            # If the region is not yet initialized, check whether it has now accumulated enough seed points 
            # reason: only after the full seeding is completed should local TuRBO start
            if not self._is_region_initialized(region_index) and X_region.shape[0] >= self.region_init_points:
                # The local TuRBO state only starts once the seed points of that egion have been evaluated, mirroring the orignal logic
                self.state.turbo_states[region_index] = self.engine.initialize_state(
                    dim=len(self.param_names),
                    batch_size=self.batch_size,
                    Y_init=Y_region,
                )
            # if it was already initialized, update the local state
            elif self._is_region_initialized(region_index):
                self.state.turbo_states[region_index] = self.engine.update_state(
                    self.state.turbo_states[region_index],
                    Y_region_new,
                )

        self.state.seen_trial_indices.update(new_trial_indices) # mark trials as processed

    @override
    def get_next_candidate(self, pending_parameters: list[TParameterization]) -> TParameterization:
        """Return one candidate to Ax, using REI/qREI seeding and TuRBO locally.

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
            If the node has not been initialized from completed trials or if
            uninitialized trust regions are still waiting for seed evaluations.
        """
        if self.parameters is None:
            raise RuntimeError("Generator state not initialized. Call update_generator_state first.")

        if self.X_hist is None or self.Y_hist is None:
            raise RuntimeError(
                "REI-TuRBO needs completed initialization trials before it can propose candidates."
            )
        # convert pending parameters into normalized tensor
        if len(pending_parameters) > 0:
            X_pending = torch.zeros(len(pending_parameters), len(self.param_names), dtype=self.dtype, device=self.device)
            for idx, pending in enumerate(pending_parameters):
                X_pending[idx, :] = torch.tensor(
                    [pending.get(name, 0.0) for name in self.param_names],
                    dtype=self.dtype,
                    device=self.device,
                )
            X_pending = self.to_unit_cube(X_pending)
        else:
            X_pending = None

        # preventing proposing duplicate candidates 
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
                # Find initialized regions where TuRBO state has triggered restart, should only happend afthe there are no pending trials
                restarted_regions = [  
                    region_index 
                    for region_index, turbo_state in self.state.turbo_states.items()
                    if turbo_state.restart_triggered and self._pending_count_for_region(region_index) == 0
                ]
                if len(restarted_regions) > 0:
                    self._restart_regions(restarted_regions) # replace restarted regions with fresh seeded ones and enqueue the new cands
                elif len(self.state.active_region_indices) == 0:
                    # When multiple trust regions are configured, qREI seeds them
                    # jointly. With a single region, the helper falls back to REI.
                    initial_regions = list(
                        range(
                            self.state.next_region_index,
                            self.state.next_region_index + self.n_trust_regions,
                        )
                    )
                    self.state.active_region_indices.extend(initial_regions)
                    self.state.next_region_index += self.n_trust_regions
                    self._enqueue_region_seed_batch(initial_regions)
                elif len(self._active_uninitialized_regions()) > 0: # if there is an actuve region that is partially seeded/not full initialized  
                    raise RuntimeError( 
                        # Prevents the generation from happening when there are still pending trials for initial seeding, making TR unusable 
                        # Current solution implements a constraint in AxBoTorchOptimizer that region_init_points must be divisible by batch_size
                        "Waiting for REI-TuRBO seed trials to complete before proposing more points."
                    ) # TODO: add a logic that would wait for it to complete
                else:
                    # Local trust-region candidate generation follows the original
                    # TuRBO-m pattern: generate from each active region, then keep
                    # only the globally best batch-sized subset.
                    self._enqueue_local_batch(X_pending=X_pending)

            item = self.state.candidate_queue.pop(0)
            params = item["params"]
            region_index = int(item["region_index"])
            key = self._key(params)
            if key in pending_keys and draws < max_draws: # if the proposed candidate is still pending in Ax, skip it and draw another one from the queue
                draws += 1
                continue
            self.state.pending_region_index_by_key[key] = region_index
            return params
