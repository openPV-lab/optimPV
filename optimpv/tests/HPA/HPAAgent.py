"""Agent wrapper for the vendored HPA benchmark problems."""

from __future__ import annotations

import numpy as np

from optimpv.general.BaseAgent import BaseAgent
from optimpv.general.general import loss_function
from optimpv.tests.HPA.HPAModel import HPAModel

# nf = number of objectives
# nx = number of free parameters
# ng = number of constraints (not supported right now)
class HPAAgent(BaseAgent):
    """Expose one HPA benchmark problem through the repository Agent interface."""

    def __init__(
        self,
        params,
        *,
        problem_name: str = "HPA101",
        n_div: int = 4,
        level: int = 0,
        normalized: bool = True,
        metric=None,
        loss="linear",
        minimize=True,
        name: str = "hpa",
        **kwargs,
    ) -> None:
        self.params = params
        self.model = HPAModel(
            problem_name=problem_name,
            n_div=n_div,
            level=level,
            normalized=normalized,
        )

        if self.model.ng > 0:
            raise ValueError(
                f"{problem_name} is an explicitly constrained HPA problem (ng={self.model.ng}). "
                "This minimal HPAAgent supports only the penalty-based unconstrained variants for now."
            )

        if metric is None:
            metric = [f"f{i+1}" for i in range(self.model.nf)]
        elif isinstance(metric, str):
            metric = [metric]
        else:
            metric = list(metric)

        if isinstance(loss, str):
            loss = [loss] * len(metric)
        else:
            loss = list(loss)

        if isinstance(minimize, bool):
            minimize = [minimize] * len(metric)
        else:
            minimize = [bool(x) for x in minimize]

        if len(metric) != self.model.nf or len(loss) != self.model.nf or len(minimize) != self.model.nf:
            raise ValueError(
                "metric, loss, and minimize must match the number of HPA objectives."
            )

        free_param_names = [p.name for p in self.params if p.type != "fixed"]
        if len(free_param_names) != self.model.nx:
            raise ValueError(
                f"{problem_name} expects {self.model.nx} free parameters, got {len(free_param_names)}."
            )

        self.param_names = free_param_names
        self.metric = metric
        self.loss = loss
        self.minimize = minimize
        self.exp_format = [problem_name] * self.model.nf
        self.name = name
        self.kwargs = kwargs

        self.all_agent_metrics = self.get_all_agent_metric_names()
        self.all_agent_tracking_metrics = None

    def run(self, parameters):
        """Evaluate the configured HPA problem and return its objective vector."""
        parameters_rescaled = self.params_rescale(parameters, self.params)
        x = np.asarray(
            [float(parameters_rescaled[name]) for name in self.param_names],
            dtype=float,
        )
        f, _ = self.model.evaluate(x) # model evaluation
        return f

    def run_Ax(self, parameters):
        """Evaluate the HPA benchmark and return Ax metric values."""
        try:
            values = self.run(parameters)
        except Exception:
            return {metric_name: np.nan for metric_name in self.all_agent_metrics}

        result = {}
        for i, metric_name in enumerate(self.all_agent_metrics):
            result[metric_name] = loss_function(float(values[i]), loss=self.loss[i])
        return result
