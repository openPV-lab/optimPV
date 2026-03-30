"""REI-TuRBO engine.

This module keeps the REI-specific trust-region logic out of the Ax node Handles both global(regional) and local optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import gpytorch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.constraints import Interval
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch.quasirandom import SobolEngine

from .rei import LogRegionalExpectedImprovement, qRegionalExpectedImprovement


@dataclass
class TurboState:
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
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


class REITuRBOEngine:
    """Node-oriented REI-TuRBO helper."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

    def initialize_state(self, dim: int, batch_size: int, Y_init: torch.Tensor) -> TurboState:
        """Create a fresh trust-region state from completed seed observations."""
        state = TurboState(dim=dim, batch_size=batch_size)
        if Y_init.numel() > 0:
            state.best_value = float(Y_init.max().item()) # take the best value from the seed data
        return state

    def update_state(self, state: TurboState, Y_next: torch.Tensor) -> TurboState:
        """Update local TuRBO trust-region state from newly completed points."""
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
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        return sobol.draw(n=n_pts).to(dtype=self.dtype, device=self.device)

    # from the orignal implementation
    def select_sample(self, X: torch.Tensor, Y: torch.Tensor, n_gp_max: int = 2000) -> torch.Tensor:
        """Subsample global history before fitting the restart-seeding GP."""
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
        """Standardize observations using the original REI-TuRBO convention."""
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
        """Generate a local TuRBO batch inside the active trust region."""
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

        if batch_size <= 1:
            ei = LogExpectedImprovement(model, Y.max())
            return optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                inequality_constraints=inequality_constraints,
            )

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
        """Fit the local TuRBO GP on the active trust-region data."""
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
        """Select fresh trust-region seed batches from the global history model."""
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

            if racqf == "QREI":
                racq_function = qRegionalExpectedImprovement(
                    X_dev=X_dev,
                    model=model,
                    best_f=train_Y.max(),
                    sampler=SobolQMCNormalSampler(sample_shape=torch.Size([rei_mc_samples])),
                    length=length_init,
                    bounds=bounds,
                )
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

            if racqf != "QREI":
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
