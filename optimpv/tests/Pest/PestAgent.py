"""Agent wrapper for the PestControl benchmark."""

import numpy as np

from optimpv.general.BaseAgent import BaseAgent
from optimpv.general.general import loss_function
from .PestModel import PestModel


class PestAgent(BaseAgent):
    """PestControl benchmark using the repository BaseAgent class."""

    def __init__(
        self,
        params,
        *,
        random_seed: int = 0,
        metric=None,
        loss="linear",
        minimize=True,
        name: str = "pest",
        **kwargs,
    ) -> None:
        self.params = params
        self.model = PestModel(random_seed=random_seed, normalize=False)

        if metric is None:
            metric = ["obj"]
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
                "metric, loss, and minimize must match the number of PestControl objectives."
            )

        free_param_names = [p.name for p in self.params if p.type != "fixed"]
        if len(free_param_names) != self.model.nx:
            raise ValueError(
                f"{self.model.problem_name} expects {self.model.nx} free parameters, got {len(free_param_names)}."
            )

        self.param_names = free_param_names
        self.metric = metric
        self.loss = loss
        self.minimize = minimize
        self.exp_format = [self.model.problem_name] * self.model.nf
        self.name = name
        self.kwargs = kwargs

        self.all_agent_metrics = self.get_all_agent_metric_names()
        self.all_agent_tracking_metrics = None

    def run(self, parameters):
        """Evaluate the configured PestControl problem and return its objective vector."""
        parameters_rescaled = self.params_rescale(parameters, self.params)
        x = np.asarray(
            [int(parameters_rescaled[name]) for name in self.param_names],
            dtype=int,
        )
        f, _ = self.model.evaluate(x)
        return f

    def run_Ax(self, parameters):
        """Evaluate the PestControl benchmark and return Ax metric values."""
        try:
            values = self.run(parameters)
        except Exception:
            return {metric_name: np.nan for metric_name in self.all_agent_metrics}

        result = {}
        for i, metric_name in enumerate(self.all_agent_metrics):
            result[metric_name] = loss_function(float(values[i]), loss=self.loss[i])
        return result
