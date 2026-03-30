"""Single-objective REI-TuRBO generation node for Ax."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, override

import numpy as np
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.parameter import RangeParameter
from ax.core.trial_status import TrialStatus
from ax.core.types import TParameterization
from ax.exceptions.core import DataRequiredError # exception when data is required, however the current node raises them directly, instead of treating it as a signal to wait
from ax.generation_strategy.external_generation_node import ExternalGenerationNode

from optimpv.optimizers.axBOtorch.rei_turbo.core import REITuRBOEngine, TurboState


@dataclass
class REITurboNodeState:
    """Mutable runtime state owned by the REI-TuRBO node."""

    turbo_states: dict[int, TurboState] = field(default_factory=dict) # state of each active trust region
    seen_trial_indices: set[int] = field(default_factory=set) # indices of trials that have been evaluated
    pending_region_index_by_key: dict[tuple[Any, ...], int] = field(default_factory=dict) # maps pending candidates to their proposed trust region
    candidate_queue: list[dict[str, Any]] = field(default_factory=list) # queue of candidates ready to be proposed to Ax
    active_region_indices: list[int] = field(default_factory=list) 
    next_region_index: int = 0


class REITuRBOGenerationNode(ExternalGenerationNode):
    """ExternalGenerationNode that uses REI to seed a TuRBO trust region."""

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

        self.engine = REITuRBOEngine(device=self.device, dtype=self.dtype)
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

    #TODO: might refactor the code to use less helper functions

    # Ax adapter logic
    def _initialize_ax_parameter_cache(self, experiment: Experiment) -> None:
        """Pull parameter metadata from Ax search space. Cache the metadata in the same order used for tensors.
        It caches parameter order, integer parameters, lower/upper bounds, normalized bounds, and parameter constraints.
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
        """Convert Ax parameter constraints to BoTorch inequality tuples."""
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
        """Stable tuple used to deduplicate queued and pending candidates."""
        key_vals: list[Any] = []
        for name in self.param_names:
            value = params[name]
            if name in self.int_param_names:
                key_vals.append(int(value))
            else:
                key_vals.append(round(float(value), self.param_key_precision))
        return tuple(key_vals)

    def _pending_count_for_region(self, region_index: int) -> int:
        return sum(1 for idx in self.state.pending_region_index_by_key.values() if idx == region_index)

    def _seed_acqf_for_batch(self, q_batch: int) -> str:
        """Decision branching point. Use qREI only for joint multi-region seeding and fall back to REI for single-region seeding."""
        if q_batch > 1:
            return "QREI"
        if self.racqf == "QREI":
            return "REI"
        return self.racqf

    def _is_region_initialized(self, region_index: int) -> bool:
        return region_index in self.state.turbo_states

    def _active_initialized_regions(self) -> list[int]:
        return [region_index for region_index in self.state.active_region_indices if self._is_region_initialized(region_index)]

    def _active_uninitialized_regions(self) -> list[int]:
        return [region_index for region_index in self.state.active_region_indices if not self._is_region_initialized(region_index)]
    # global
    def _append_history(self, X_new: torch.Tensor, Y_new: torch.Tensor) -> None:
        """Append newly completed trials to the global history used for REI restart seeding."""
        if self.X_hist is None or self.Y_hist is None:
            self.X_hist = X_new
            self.Y_hist = Y_new
        else:
            self.X_hist = torch.cat((self.X_hist, X_new), dim=0)
            self.Y_hist = torch.cat((self.Y_hist, Y_new), dim=0)
    # local specific to each trust region
    def _append_region_data(self, region_index: int, X_new: torch.Tensor, Y_new: torch.Tensor) -> None:
        """Append newly completed trials belonging to one active trust region."""
        if region_index not in self.X_turbo_by_region or region_index not in self.Y_turbo_by_region:
            self.X_turbo_by_region[region_index] = X_new
            self.Y_turbo_by_region[region_index] = Y_new
        else:
            self.X_turbo_by_region[region_index] = torch.cat((self.X_turbo_by_region[region_index], X_new), dim=0)
            self.Y_turbo_by_region[region_index] = torch.cat((self.Y_turbo_by_region[region_index], Y_new), dim=0)
    # corresponds to orginal turbo-m behaviour where restarted region is replaced by a fresh ona and (tr_active is updated)
    def _remove_region(self, region_index: int) -> None:
        """Drop a finished trust region before replacing it with a fresh seeded region."""
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
        Pools all candidates across regions and keeps the globally best amount basedon barch_size 
        """
        if self.bounds_tensor is None:
            raise RuntimeError("Bounds not initialized.")

        candidate_items: list[dict[str, Any]] = []
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
            acq_flat = acq_value.reshape(-1)
            for row_idx, candidate in enumerate(X_next_un):
                params: dict[str, Any] = {}
                for idx, name in enumerate(self.param_names):
                    raw_value = float(candidate[idx].item())
                    params[name] = int(round(raw_value)) if name in self.int_param_names else raw_value
                candidate_items.append(
                    {
                        "params": params,
                        "region_index": region_index,
                        "score": float(acq_flat[row_idx].item()),
                    }
                )

        if len(candidate_items) == 0:
            raise RuntimeError("No initialized REI-TuRBO regions are ready to generate local candidates.")

        candidate_items.sort(key=lambda item: item["score"], reverse=True)
        for item in candidate_items[: self.batch_size]:
            self.state.candidate_queue.append(
                {
                    "params": item["params"],
                    "region_index": item["region_index"],
                }
            )

    # logic similar to orignal turbo-m restart where restarted regions are replaced by fresh seeded regions from the global model.
    # exception is that with the Ax adapter, the restart trigger is only checked when the candidate queue is empty 
    # and new candidates need to be generated. 
    def _restart_regions(self, region_indices: list[int]) -> None:
        """Replace restarted trust regions with freshly seeded regions from the global history with new iDs.
        """
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
        if self.bounds_tensor is None:
            raise RuntimeError("Bounds not initialized.")
        return (X - self.bounds_tensor[0]) / (self.bounds_tensor[1] - self.bounds_tensor[0])

    def from_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        if self.bounds_tensor is None:
            raise RuntimeError("Bounds not initialized.")
        return X * (self.bounds_tensor[1] - self.bounds_tensor[0]) + self.bounds_tensor[0]

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """Ingest newly completed Ax trials and update global/local REI-TuRBO state."""
        if self.parameters is None: # initialize on first call
            self._initialize_ax_parameter_cache(experiment=experiment)

        if experiment.optimization_config is None:
            raise ValueError("Experiment has no optimization_config configured.")
        metric_names = list(experiment.optimization_config.metrics.keys())
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
        """Return one candidate to Ax, using REI/qREI seeding and TuRBO locally."""
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
                    raise RuntimeError( # here is where the error is raised meaning there is at least one region that is has not a fully initialized state
                        # This error is needed as it prevents the generation from happening when there are still pending trials for initial seeding, making TR unusable 
                        "Waiting for REI-TuRBO seed trials to complete before proposing more points."
                    ) # TODO: add a logic that would wait for it to complete, maybe the logic in AxBotorchOptimize could have a flag that makes it wait and call get_next_candidate after the seeding trials are done.
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
