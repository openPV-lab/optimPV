"""UltraNestOptimizer module.

This module contains the UltraNestOptimizer class. The class is used to run
Bayesian inference using the UltraNest nested sampling library.
"""
######### Package Imports #########################################################################
# installed ultranest & corner
import os
import warnings
from functools import partial
from logging import Logger

import corner
import matplotlib.pyplot as plt
import numpy as np

try:
    from ultranest import ReactiveNestedSampler
except ImportError:  # pragma: no cover - handled at runtime when dependency is absent
    ReactiveNestedSampler = None

from optimpv.general.BaseAgent import BaseAgent
from optimpv.general.logger import get_logger, _round_floats_for_logging

# Logger setup
logger: Logger = get_logger("UltraNestOptimizer")
ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)


######### Optimizer Definition #######################################################################
class UltraNestOptimizer(BaseAgent):
    """
    Optimizer using the UltraNest library for nested-sampling Bayesian inference.
    Inherits from BaseAgent and interacts with Agent objects.
    """

    def __init__(
        self,
        params=None,
        agents=None,
        progress=True,
        log_dir=None,
        save_logs=False,
        resume="subfolder",
        name="ultranest",
        **kwargs,
    ):
        """_summary_

        Parameters
        ----------
        params : list of Fitparam() objects, optional
            List of Fitparam() objects, by default None
        agents : list of Agent() objects, optional
            List of Agent() objects see optimpv/general/BaseAgent.py for a base class definition, by default None
        progress : bool, optional
            Whether to display sampler progress information, by default True
        log_dir : str or None, optional
            Directory used by UltraNest to store run outputs. If None and `save_logs` is True,
            logs are stored under ``optimpv/optimizers/BayesInfUltraNest/ultranest_logs/<name>``,
            by default None
        save_logs : bool, optional
            Whether UltraNest should write run outputs to disk, by default False
        resume : str, optional
            UltraNest resume mode, by default "subfolder"
        name : str, optional
            Name for the inference process, by default "ultranest"
        **kwargs : dict, optional
            Additional keyword arguments such as sampler_kwargs, run_kwargs, and logging settings, by default None

        Raises
        ------
        ValueError
            Agents must minimize all targets. Please set minimize=True for all targets.
        ValueError
            Parameter must be of type 'float'. Please set value_type='float' for all parameters.
        ValueError
            Number of dimensions (parameters) cannot be determined.
        """
        # super().__init__() # Call BaseAgent init if needed
        self.params = params
        if not isinstance(agents, list):
            agents = [agents]
        self.agents = agents
        self.progress = progress
        self.log_dir = log_dir
        self.save_logs = bool(save_logs)
        self.resume = resume
        self.name = name
        self.kwargs = kwargs

        # make sure all agents target are minimize
        for agent in self.agents:
            if hasattr(agent, "minimize"):
                for i in range(len(agent.minimize)):
                    if not agent.minimize[i]:
                        raise ValueError(
                            f"Agent {agent.name} must minimize all targets. Please set minimize=True for all targets."
                        )
        # make sure all of the params val_type are floats
        for param in self.params:
            if param.value_type != "float":
                raise ValueError(
                    f"Parameter {param.name} must be of type 'float'. Please set value_type='float' for all parameters."
                )

        # Process parameters to get dimensions, bounds, initial guess, names
        self.x0, self.bounds, self.param_mapping, self.log_params_indices = self.create_search_space(self.params)
        self.ndim = len(self.x0)
        
        self.param_names = [
            p.display_name if hasattr(p, "display_name") else p.name
            for p in self.params
            if p.name in self.param_mapping
        ]

        if self.ndim == 0:
            raise ValueError("Number of dimensions (parameters) cannot be determined.")

        default_log_root = os.path.join(os.path.dirname(__file__), "ultranest_logs")
        default_log_dir = os.path.join(default_log_root, self.name)
        if self.save_logs:
            self.log_dir = os.path.abspath(self.log_dir or default_log_dir)
        else:
            self.log_dir = None
        self.sampler_kwargs = dict(self.kwargs.get("sampler_kwargs", {}))
        self.run_kwargs = dict(self.kwargs.get("run_kwargs", {}))

        if self.progress and "show_status" not in self.run_kwargs:
            self.run_kwargs["show_status"] = True

        self.sampler = None
        self.chain = None
        self.flat_samples = None
        self.flat_samples_orig = None
        self.log_prob_samples = None
        self.results = None
        self.nested_result = None
        self.all_metrics = self.create_metrics_list()

    def create_metrics_list(self):
        """Create a list of all metrics from all agents.

        Returns
        -------
        list
            List of metric names
        """
        metrics = []
        for agent in self.agents:
            for i in range(len(agent.all_agent_metrics)):
                metrics.append(agent.all_agent_metrics[i])
        return metrics

    def create_search_space(self, params):
        """Create search space details (initial vector, bounds, mapping) from FitParam list.

        Parameters
        ----------
        params : list of FitParam
            List of FitParam objects defining the parameters to optimize.

        Returns
        -------
        tuple
            x0 : array
                Initial parameter vector for optimization, potentially in log10 space.
            bounds : list of tuples
                List of ``(lower, upper)`` bound tuples for the optimization vector.
            param_mapping : list
                List of parameter names corresponding to the entries in ``x0``.
            log_params_indices : list
                Indices of parameters optimized in log10 space.

        Raises
        ------
        ValueError
            If a parameter type is unsupported (not 'float').
        """
        # Initialize empty lists for x0, bounds, and parameter mapping
        x0 = []
        bounds = []
        param_mapping = []
        log_params_indices = []

        for param in params:
            if param.type == "fixed":
                continue

            param_mapping.append(param.name)
            current_index = len(x0) # Index in the optimization vector 'x'

            if param.value_type == "float":
                if param.force_log:
                    log_params_indices.append(current_index)
                    x0.append(np.log10(param.value))
                    # Ensure bounds are positive before log10
                    lower_bound = np.log10(param.bounds[0]) if param.bounds[0] > 0 else -np.inf
                    upper_bound = np.log10(param.bounds[1]) if param.bounds[1] > 0 else np.inf
                    bounds.append((lower_bound, upper_bound))
                else:
                    scale_factor = (
                        param.fscale
                        if hasattr(param, "fscale") and param.fscale is not None
                        else 1.0
                    )
                    x0.append(param.value / scale_factor)
                    bounds.append((param.bounds[0] / scale_factor, param.bounds[1] / scale_factor))
            else:
                raise ValueError(
                    f"Unsupported parameter type: {param.value_type}. Only 'float' is supported."
                )

        return np.array(x0), bounds, param_mapping, log_params_indices

    def reconstruct_params(self, x_opt):
        """Reconstruct a full parameter dictionary from an optimization vector.

        Parameters
        ----------
        x_opt : array-like
            Parameter vector from the sampler, potentially log-transformed.

        Returns
        -------
        dict
            Dictionary mapping full parameter names to their values.

        Raises
        ------
        ValueError
            If a parameter type is unsupported (not 'float').
        """
        # Initialize empty dictionary for reconstructed parameters
        param_dict = {}
        opt_idx = 0

        for param in self.params:
            if param.type == "fixed":
                param_dict[param.name] = param.value
            else:
                # Find the corresponding value in x_opt
                current_val = x_opt[opt_idx]

                if param.value_type == "float":
                    if opt_idx in self.log_params_indices:
                        if param.force_log:
                            # If log10 transformed, convert back to original scale
                            param_dict[param.name] = 10**current_val
                        else:
                            param_dict[param.name] = current_val
                    else:
                        scale_factor = (
                            param.fscale
                            if hasattr(param, "fscale") and param.fscale is not None
                            else 1.0
                        )
                        param_dict[param.name] = current_val * scale_factor
                else:
                    raise ValueError(
                        f"Unsupported parameter type: {param.value_type}. Only 'float' is supported."
                    )

                opt_idx += 1

        return param_dict

    def _log_likelihood(self, theta, agents=None):
        """Calculate the log-likelihood based on agent evaluations.
        Assumes agent.run_Ax returns a dictionary where keys match
        self.all_metrics and values are loss or metric values.  
        Converts loss to log-likelihood assuming Gaussian noise.

        Parameters
        ----------
        theta : array-like
            Contains the parameters to evaluate.
        agents : list of Agent() objects, optional
            List of Agent() objects to evaluate the likelihood, by default None

        Returns
        -------
        float
            Log-likelihood value. Returns -np.inf for invalid evaluations such as NaN or Inf values.
        """
        theta = np.asarray(theta)
        param_dict = {}
        idx = 0
        for i, param in enumerate(self.params):
            if param.type == 'fixed':
                param_dict[param.name] = param.value
            else:
                param_dict[param.name] = theta[idx]
                idx += 1

        total_log_like = 0.0
        all_results = {}

        try:
            for agent in agents:
                agent_results = agent.run_Ax(param_dict)
                all_results.update(agent_results)

            for metric_name in self.all_metrics:
                if metric_name in all_results:
                    loss_val = all_results[metric_name]
                    if np.isnan(loss_val) or not np.isfinite(loss_val): # return -inf for any invalid vals
                        return -np.inf
                    total_log_like += -0.5 * loss_val
                else:
                    warnings.warn(
                        f"Metric {metric_name} not found in agent results for params {param_dict}, something went wrong."
                    )
                    return -np.inf

            # safety checks
            if not np.isfinite(total_log_like):
                return -np.inf

            return total_log_like
        except Exception:
            return -np.inf

    # ultranest helper function
    def _prior_transform(self, cube):
        """Transform a unit-cube sample into the optimization space parameter vector.

        Parameters
        ----------
        cube : array-like
            Unit-cube sample provided by UltraNest.

        Returns
        -------
        np.ndarray
            Transformed parameter vector in optimization space.
        """
        cube = np.asarray(cube)
        theta = np.empty(self.ndim, dtype=float)
        for i, (lower, upper) in enumerate(self.bounds):
            theta[i] = lower + cube[i] * (upper - lower)
        return theta

    # ultranest helper function
    def _samples_to_original_space(self, samples_optimization):
        """Convert optimization space samples to original parameter space.

        Parameters
        ----------
        samples_optimization : array
            Samples in optimization space.

        Returns
        -------
        np.ndarray
            Samples transformed back to the original parameter space.
        """
        samples_optimization = np.asarray(samples_optimization)
        samples_original = np.empty_like(samples_optimization, dtype=float)

        for i, name in enumerate(self.param_mapping):
            original_param = next(p for p in self.params if p.name == name)
            if i in self.log_params_indices: # if log10 transformed, convert back to original scale
                samples_original[:, i] = 10 ** samples_optimization[:, i]
            else:
                scale_factor = ( # otherwise apply fscale if defined
                    original_param.fscale
                    if hasattr(original_param, "fscale") and original_param.fscale is not None
                    else 1.0
                )
                samples_original[:, i] = samples_optimization[:, i] * scale_factor

        return samples_original

    def optimize(self):
        """Run the nested sampling optimization using UltraNest.

        Returns
        -------
        dict
            Dictionary containing posterior summary statistics for each fitted parameter.

        Raises
        ------
        ImportError
            If the ``ultranest`` package is not installed.
        """

        verbose_logging = self.kwargs.get("verbose_logging", True)

        if verbose_logging:
            print("----------------------------------------------------\n")
            logger.info("Running UltraNest nested sampling...")

        def loglike(theta):
            return self._log_likelihood(theta, agents=self.agents)

        sampler = ReactiveNestedSampler(
            self.param_mapping,
            loglike,
            transform=self._prior_transform,
            log_dir=self.log_dir, # for logging
            resume=self.resume, # resume/subfolder/overwrite behavior
            **self.sampler_kwargs,
        )
        result = sampler.run(**self.run_kwargs) # execute the sampler and pass any additional run kwargs

        self.sampler = sampler
        self.nested_result = result

        if verbose_logging:
            logger.info("UltraNest run complete.")

        self.flat_samples = np.asarray(result["samples"])
        self.flat_samples_orig = self._samples_to_original_space(self.flat_samples)
        self.log_prob_samples = np.asarray(result["weighted_samples"]["logl"])
        self.chain = None

        self.results = {}
        # Calculate percentiles in original space
        for i, name in enumerate(self.param_mapping):
            param_samples_orig = self.flat_samples_orig[:, i]
            mcmc = np.percentile(param_samples_orig, [16, 50, 84])
            q = np.diff(mcmc)
            self.results[name] = {"median": mcmc[1], "16th": mcmc[0], "84th": mcmc[2], "lower_err": q[0], "upper_err": q[1],}
            display_name = next(
                (p.display_name for p in self.params if p.name == name and hasattr(p, "display_name")),
                name,
            )
            if verbose_logging:
                # Log results
                logger.info(
                    f"{display_name} ({name}): {mcmc[1]:.4g} (+{q[1]:.3g} / -{q[0]:.3g})"
                )
        # Update self.params with median values
        self.update_params_with_best_balance()
        if verbose_logging:
            print("----------------------------------------------------\n")
        return self.results
    def _original_vector_to_param_dict(self, x_orig):
        param_dict = {}
        idx = 0
        for param in self.params:
            if param.type == "fixed":
                param_dict[param.name] = param.value
            else:
                param_dict[param.name] = x_orig[idx]
                idx += 1
        return param_dict

    def get_best_params(self, method="max_likelihood"):
        if self.flat_samples is None:
            print("Optimization has not been run yet.")
            return None

        if method == "median":
            best_orig = np.median(self.flat_samples_orig, axis=0)
            return self._original_vector_to_param_dict(best_orig)

        if method == "mean":
            best_orig = np.mean(self.flat_samples_orig, axis=0)
            return self._original_vector_to_param_dict(best_orig)

        if method == "max_likelihood":
            if self.nested_result is None:
                raise ValueError("Optimization has not run or results not processed.")
            best_point_opt = np.asarray(self.nested_result["maximum_likelihood"]["point"], dtype=float)
            best_orig = self._samples_to_original_space(best_point_opt[np.newaxis, :])[0]
            return self._original_vector_to_param_dict(best_orig)

        raise ValueError("Method must be 'median', 'mean', or 'max_likelihood'")
    # def get_best_params(self, method="max_likelihood"):
    #     """Return the 'best' parameters based on the nested-sampling samples.
    #     This method allows the user to specify how to determine the 'best' parameters

    #     Parameters
    #     ----------
    #     method : str
    #         How to determine 'best' params ('median', 'mean', 'max_likelihood').
    #         'median' - median of the samples
    #         'mean' - mean of the samples
    #         'max_likelihood' - maximum likelihood estimate based on the UltraNest result
    #         'max_likelihood' is the default method.

    #     Returns
    #     -------
    #     dict
    #         Dictionary of best parameter values in original space.

    #     Raises
    #     ------
    #     ValueError
    #         If the method is not one of 'median', 'mean', or 'max_likelihood'.
    #     """
    #     if self.flat_samples is None:
    #         print("Optimization has not been run yet.")
    #         return None

    #     if method == "median":
    #         best_opt_params = np.median(self.flat_samples, axis=0)
    #         return self.reconstruct_params(best_opt_params)
    #     if method == "mean":
    #         best_opt_params = np.mean(self.flat_samples, axis=0)
    #         return self.reconstruct_params(best_opt_params)
    #     if method == "max_likelihood":
    #         if self.nested_result is None:
    #             raise ValueError("Optimization has not run or results not processed.")
    #         best_point = self.nested_result["maximum_likelihood"]["point"]
    #         return self.reconstruct_params(np.asarray(best_point))

    #     raise ValueError("Method must be 'median', 'mean', or 'max_likelihood'")

    def update_params_with_best_balance(self, method="max_likelihood", return_best_balance=False):
        """Update the parameters with the best values based on nested-sampling results.
        This method updates the parameters in self.params with the best values
        determined by the specified method. It can also return the best parameters
        dictionary if requested.

        Parameters
        ----------
        method : str, optional
            Method to determine 'best' params ('median', 'mean', 'max_likelihood'), by default 'max_likelihood'
        return_best_balance : bool, optional
            If True, return the best parameters dictionary, by default False

        Returns
        -------
        dict or None
            Dictionary of best parameter values if ``return_best_balance`` is True, otherwise None.

        Raises
        ------
        ValueError
            If optimization has not run or results were not processed.
        """
        if self.results is None:
            raise ValueError("Optimization has not run or results not processed.")

        best_params_dict = self.get_best_params(method=method)

        # Update the FitParam objects in self.params
        for param in self.params:
            if param.name in best_params_dict:
                param.value = best_params_dict[param.name]

        if return_best_balance:
            return best_params_dict # Return the dictionary used for updating

    def get_chain(self):
        """Return the stored chain samples."""
        if self.flat_samples is not None:
            return self.flat_samples
        return None

    def get_flat_samples(self):
        """Return the equally weighted posterior samples in optimization space."""
        return self.flat_samples

    def get_flat_samples_original(self):
        """Return the equally weighted posterior samples in original parameter space."""
        return self.flat_samples_orig

    def plot_corner(self, **kwargs):
        """Generate a corner plot of the posterior distribution.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to ``corner.corner``.

        Returns
        -------
        matplotlib.figure.Figure or None
            Corner plot figure if samples are available, otherwise None.
        """
        title_fmt = kwargs.get("title_fmt", ".4e")
        if self.flat_samples_orig is None:
            print("Optimization has not been run yet.")
            return None

        labels_orig = []
        truths_orig = kwargs.get("True_params", None)

        for name in self.param_mapping:
            original_param = next(p for p in self.params if p.name == name)
            labels_orig.append(
                original_param.display_name if hasattr(original_param, "display_name") else name
            )

        if truths_orig is None:
            truths_list = [None] * len(self.param_mapping)
        else:
            truths_list = [truths_orig.get(name, None) for name in self.param_mapping]

        corner_kwargs = {
            "labels": labels_orig,
            "show_titles": True,
            "title_kwargs": {"fontsize": 10},
            "quantiles": [0.16, 0.5, 0.84],
            "truths": truths_list,
            "truth_color": "red",
            "color": "darkblue",
            "hist2d_kwargs": {"cmap": plt.get_cmap("Blues")},
            "hist_kwargs": {"color": "darkblue"},
        }
        corner_kwargs.update(kwargs)

        params_axis_type = []
        for param in self.params:
            if hasattr(param, "axis_type"):
                params_axis_type.append(param.axis_type)
            else:
                params_axis_type.append("linear")

        fig = corner.corner(
            self.flat_samples_orig,
            axes_scale=params_axis_type,
            title_fmt=title_fmt,
            **corner_kwargs,
        )
        return fig

    def plot_run(self, **kwargs):
        """Call the UltraNest plotting helper when available.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to ``self.sampler.plot``.

        Returns
        -------
        object or None
            Result returned by the UltraNest plotting helper if available, otherwise None.
        """
        if self.sampler is None:
            print("Optimization has not been run yet.")
            return None
        return self.sampler.plot(**kwargs)
