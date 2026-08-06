"""Stage 3 of the combinatorial-sputtering -> device-physics -> ML pipeline.

MLSurrogateAgent wraps a fitted Stage 2 model (ProcessCorrelationModel, or any object
exposing .predict(df) -> DataFrame of target columns) as an optimpv Agent, so it can be
handed to axBOtorchOptimizer exactly like JVAgent/DiodeAgent -- except each "simulation"
is an instant regressor prediction instead of an external SIMsalabim call. This is what
turns Stage 2's process<->performance correlation into an actual search: propose/predict
the ETL/HTL/PVK/TCO process conditions (or thicknesses) expected to maximize PCE (or
whichever targets you trained on).
"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd

from optimpv.general.BaseAgent import BaseAgent

######### Agent Definition #######################################################################


class MLSurrogateAgent(BaseAgent):
    """Agent that evaluates a trained ML surrogate instead of running a simulator.

    Parameters
    ----------
    params : list of Fitparam() objects
        The process-condition parameters being optimized (e.g. RF power, pressure,
        O2:Ar ratio, thickness for ETL/HTL/TCO/PVK) -- these must match
        `model.feature_cols`.
    model : object
        A fitted model exposing .predict(df) -> DataFrame with one column per target
        (e.g. a fitted optimpv.ml.ProcessCorrelationModel). Column names must match
        `targets`.
    targets : list of str
        Which of the model's output columns to expose as optimization objectives.
    minimize : bool or list of bool, optional
        Whether each target should be minimized (e.g. a defect density) or maximized
        (e.g. PCE), by default False (maximize) for every target.
    name : str, optional
        Name of the agent, by default 'ml_surrogate'.
    **kwargs : dict
        Additional keyword arguments (stored, unused by this agent).
    """

    def __init__(self, params, model, targets, minimize=False, name='ml_surrogate', **kwargs):
        self.params = params
        self.model = model
        self.name = name
        self.kwargs = kwargs

        self.exp_format = list(targets)
        self.metric = [None] * len(self.exp_format)
        self.loss = [None] * len(self.exp_format)
        self.threshold = [None] * len(self.exp_format)
        self.minimize = [minimize] * len(self.exp_format) if isinstance(minimize, bool) else list(minimize)
        if len(self.minimize) != len(self.exp_format):
            raise ValueError('minimize must be a bool or a list the same length as targets')

        self.tracking_metric = None
        self.tracking_loss = None
        self.tracking_exp_format = None

        self.all_agent_metrics = self.exp_format  # one metric per target, named after the target itself

    def run(self, parameters):
        """Predict the target(s) for a given process-condition point.

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names (matching self.model.feature_cols) and
            values. If empty, uses the current self.params values (mirrors the
            JVAgent/DiodeAgent convention of `run(parameters={})` after fitting).

        Returns
        -------
        dict
            {target_name: predicted_value}
        """
        if parameters:
            row = {}
            for p in self.params:
                row[p.name] = parameters.get(p.name, p.value)
        else:
            row = {p.name: p.value for p in self.params}

        df = pd.DataFrame([row])
        pred = self.model.predict(df)
        return {t: float(pred[t].iloc[0]) for t in self.exp_format}

    def run_Ax(self, parameters):
        """Return the predicted target(s), keyed by metric name, for Ax/BoTorch.

        Note there is no loss/metric transform here (unlike JVAgent/DiodeAgent, which
        compare a simulation to measured data): the surrogate's raw prediction *is*
        the objective, and `self.minimize` controls whether axBOtorchOptimizer treats
        it as something to minimize or maximize.
        """
        preds = self.run(parameters)
        return {self.all_agent_metrics[i]: preds[t] for i, t in enumerate(self.exp_format)}
