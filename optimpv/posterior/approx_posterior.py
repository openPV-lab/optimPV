"""Module containing classes and functions for posterior analysis of parameters using the ML models from the BO optimization.
This module provides functionality to visualize the posterior distributions of parameters
using various plots, including 1D and 2D posteriors, devil's plots, and density plots."""

######### Package Imports #########################################################################
import torch, copy, itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.special import logsumexp
from torch import logsumexp as torch_logsumexp
from functools import partial
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import Normalize  
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import arviz as az

# optimpv
from optimpv.general.BaseAgent import BaseAgent
import logging
from logging import Logger
from ax.utils.common.logger import set_ax_logger_levels
set_ax_logger_levels(logging.WARN)
from optimpv.general.logger import get_logger, _round_floats_for_logging

logger: Logger = get_logger('approx_posterior')
ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)

torch.set_default_dtype(torch.float64)


######### Function Definitions ####################################################################

class ApproximatePosterior(BaseAgent):
    def __init__(self, params, df, sigma, outcome_name, model=None, best_params = None, is_nat_scale = False, **kwargs):
        """ Class to handle the approximate posterior using a surrogate model trained on the optimization data.

        Parameters
        ----------
        params : list of Fitparam() objects
            list of Fitparam() objects defining the parameters to optimize
        df : pd.DataFrame
            DataFrame containing the data used for training the surrogate model
        sigma : float or torch.Tensor
            Noise level or uncertainty associated with the observations
        outcome_name : str
            Name of the outcome variable in the DataFrame
        model : BotorchModel, optional
            Surrogate model used for the approximate posterior, by default None
        best_params : dict, optional
            Dictionary of best parameter values, by default None
        is_nat_scale : bool, optional
            Indicates if the parameters are in natural scale, by default False
        """          
        
        self.params = params
        params_names = [param.name for param in params if not param.type == 'fixed']
        self.sigma = sigma
        self.outcome_name = outcome_name
        self.model = model
        self.kwargs = kwargs
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.name_free_params = [p.name for p in self.params if not p.type == 'fixed']
        self.num_free_params = len(self.name_free_params)

        if best_params is None:
            best_idx = df[outcome_name].idxmax()
            best_params = df.loc[best_idx][params_names].to_dict()

        # get the dataframe in the rescaled space and keep the true dataframe, df and best_params are always rescaled and will be used for training the surrogate model
        if is_nat_scale:
            self.true_df = df
            self.df = self.descale_dataframe(copy.deepcopy(df), params)
            self.true_best_params = best_params
            self.best_params = self.params_descale(best_params, params)
        else:
            self.true_df = self.rescale_dataframe(copy.deepcopy(df), params)
            self.df = df
            self.true_best_params = self.params_rescale(best_params, params)
            self.best_params = best_params

        # take_log_params
        take_log_params_names = [] # list of parameter names to take log
        for p in self.params:
            if p.force_log == True: # the log trasnform was already applied to the dataframe and best_params
                take_log_params_names.append(p.name)
            else:
                if p.log_scale == True : # here we do the log transform of the dataframe and best_params to imporve the GP model training
                    self.df[p.name] = np.log10(self.df[p.name].values)
                    self.best_params[p.name] = np.log10(self.best_params[p.name])
                    take_log_params_names.append(p.name)
        self.take_log_params_names = take_log_params_names
       
        max_size_cv = kwargs.get('max_size_cv', 300)
        if len(self.df) > max_size_cv:
            self.do_batch_cv = False
        else:
            self.do_batch_cv = True
        
        # convert sigma to tensor
        if not isinstance(sigma, torch.Tensor) and sigma.device != self.device:
            self.sigma = torch.tensor(sigma).to(self.device)
        
        # get bounds tensor
        true_bounds_tensor = self.get_bounds_list()
        self.bounds_tensor = torch.tensor(self.bounds_descale(true_bounds_tensor, params)).to(self.device)
        self.true_bounds_tensor = torch.tensor(true_bounds_tensor).to(self.device)

        self.log10_transform_LLH = kwargs.get('log10_transform_LLH', True) # whether to log10-transform the likelihood values for the GPR model, i.e. the GPR model predicts log(-LLH) instead of LLH directly
        self.model = model
        self.X, self.y = self.get_Xy_train_tensor()
        

    
    def get_Xy_train_tensor(self):
        """Get the training data as tensors.

        Returns
        -------
        X_train : torch.Tensor
            Tensor containing the training input data.
        y_train : torch.Tensor
            Tensor containing the training output data.
        """
        
        X_train = []
        y_train = []
        for index, row in self.df.iterrows():
            x_row = []
            for p in self.params:
                if not p.type == 'fixed':
                    x_row.append(row[p.name])
            X_train.append(x_row)
            y_train.append(row[self.outcome_name])

        X_train = torch.tensor(X_train).to(self.device).reshape(-1, self.num_free_params)
        y_train = torch.tensor(y_train).to(self.device).unsqueeze(-1)
        
        self.X = X_train
        self.y = y_train
        return X_train, y_train

    def train_surrogate_model(self):
        """Train the surrogate model using the training data.
        Add to self the trained model and cross-validation results if applicable.
        """        

        # get the data from the dataframe
        df = self.df
        outcome_name = self.outcome_name

        # get the training data
        if self.X is not None and self.y is not None:
            X_tensor = self.X
            y_tensor = self.y
        else:
            X_tensor, y_tensor = self.get_Xy_train_tensor()

        if self.log10_transform_LLH:
            # check if all y_train values are negative or positives
            if not torch.all((y_tensor * y_tensor) > 0):
                logger.warning(r"/!\\ The log10 transform is used but not all outcome values are strictly positive or negative. Consider revising the data or transformation.")
               
            y_tensor = torch.log10(y_tensor.abs())

        input_transform = Normalize(d=X_tensor.shape[-1], bounds=self.bounds_tensor)

        if self.do_batch_cv:
            cv_folds = gen_loo_cv_folds(X_tensor, y_tensor)

            outcome_transform = Standardize(m=1)
            outcome_transform._batch_shape = cv_folds.train_Y.shape[:-2]

            # composite_kernel = ProductKernel(
            #     MaternKernel(nu=2.5, ard_num_dims=self.num_free_params),
            #      LinearKernel(ard_num_dims=self.num_free_params)
            # )
            # composite_kernel = ProductKernel(
            #     MaternKernel(nu=2.5, ard_num_dims=self.num_free_params),
            #      PolynomialKernel(power=2, ard_num_dims=self.num_free_params)
            # )
            # Perform batch cross-validation
            cv_results = batch_cross_validation(
                model_cls= SingleTaskGP,
                mll_cls=ExactMarginalLogLikelihood,
                cv_folds=cv_folds,
                model_init_kwargs={
                    "input_transform": input_transform,
                    "outcome_transform": outcome_transform,
                    "covar_module": ScaleKernel(base_kernel=MaternKernel(nu=2.5, ard_num_dims=self.num_free_params)),
                    # "covar_module": ScaleKernel(composite_kernel),
                    "likelihood": GaussianLikelihood(),
                }
            )
            self.cv_results = cv_results
            self.cv_folds = cv_folds
            self.model = cv_results.model
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_tensor,   y_tensor, test_size=0.1)
            # Fit the model on the entire dataset
            gp_model = SingleTaskGP(
                train_X=X_train,
                train_Y=y_train,
                input_transform=input_transform,
                outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(base_kernel=MaternKernel(nu=2.5, ard_num_dims=self.num_free_params)),
                # covar_module=ScaleKernel(composite_kernel),
                likelihood=GaussianLikelihood(),
            ).to(self.device)

            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model).to(self.device)
            fit_gpytorch_mll(mll)

            self.cv_results = None
            self.cv_folds = None
            self.model = gp_model
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

    def plot_cv_results(self, ax=None, **kwargs):
        """Plot the cross-validation results into a provided Axes (or create one).

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Existing axes to draw into. If None, a new figure/axes are created internally, by default None
        kwargs : dict
            Plot customization options:
              - figsize: tuple, size when creating a new axes (default (4, 4))
              - xlabel: str, x-axis label (default "Actual")
              - ylabel: str, y-axis label (default "Predicted")
              - fmt: str, marker format for errorbar (default "*")
              - color: str, color for points (passed to errorbar)
              - errorbar_kwargs: dict, extra kwargs forwarded to ax.errorbar
              - diag_color: str, color for parity diagonal (default "k")
              - diag_ls: str, linestyle for parity diagonal (default "-")
              - diag_lw: float, linewidth for parity diagonal (default 2)
              - diag_label: str, label for parity diagonal (default "true objective")
              - title: str, custom title. If not provided, shows R2 and CV Error.
        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        # Ensure CV results exist (this function assumes batch CV was run)
        if not hasattr(self, 'cv_results') or self.cv_results is None:
            self.train_surrogate_model()

        posterior = self.cv_results.posterior
        mean = posterior.mean
        cv_error = ((self.cv_folds.test_Y.squeeze() - mean.squeeze()) ** 2).mean()
        
        # Confidence intervals
        lower, upper = posterior.mvn.confidence_region()

        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', (4, 4)))

        # Prepare data
        y_true = self.cv_folds.test_Y.cpu().numpy().flatten()
        y_pred = mean.cpu().numpy().flatten()
        y_err = ((upper - lower) / 2).cpu().numpy().flatten()

        # Parity diagonal
        diag_min = y_true.min()
        diag_max = y_true.max()
        ax.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            kwargs.get('diag_ls', '-'),
            color=kwargs.get('diag_color', 'k'),
            linewidth=kwargs.get('diag_lw', 2),
            label=kwargs.get('diag_label', 'true objective'),
        )

        ax.set_xlabel(kwargs.get('xlabel', 'Actual'))
        ax.set_ylabel(kwargs.get('ylabel', 'Predicted'))

        # Errorbar point style
        fmt = kwargs.get('fmt', '*')
        err_kwargs = kwargs.get('errorbar_kwargs', {}).copy()
        if 'color' not in err_kwargs and 'color' in kwargs:
            err_kwargs['color'] = kwargs['color']

        ax.errorbar(x=y_true, y=y_pred, yerr=y_err, fmt=fmt, **err_kwargs)

        # Metrics and title
        self.r2 = r2_score(y_true.reshape(-1), y_pred)
        title = kwargs.get('title', f"R2: {self.r2:.2f}, CV Error: {cv_error:.3f}")
        ax.set_title(title)

        return ax

    def plot_parity(self, ax=None, **kwargs):
        """Plot the parity plot of the surrogate model into a provided Axes (or create one).

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Existing axes to draw into. If None, a new figure/axes are created internally.
        kwargs : dict
            Plot customization options:
              - figsize: tuple, size when creating a new axes (default (4, 4))
              - xlabel: str, x-axis label (default "Actual")
              - ylabel: str, y-axis label (default "Predicted")
              - fmt: str, marker format for errorbar (default "*")
              - color: str, color for points (passed to errorbar)
              - errorbar_kwargs: dict, extra kwargs forwarded to ax.errorbar
              - diag_color: str, color for parity diagonal (default "k")
              - diag_ls: str, linestyle for parity diagonal (default "-")
              - diag_lw: float, linewidth for parity diagonal (default 2)
              - diag_label: str, label for parity diagonal (default None, no legend)
              - title: str, custom title. If not provided, shows R2 and MSE.
        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if self.do_batch_cv:
            return self.plot_cv_results(ax=ax, **kwargs)

        if self.model is None:
            self.train_surrogate_model()

        # Predict on test set
        posterior = self.model.posterior(self.X_test)
        mean = posterior.mean
        lower, upper = posterior.mvn.confidence_region()
        test_error = ((self.y_test.squeeze() - mean.squeeze()) ** 2).mean()  # MSE

        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', (4, 4)))

        # Prepare data
        y_true = self.y_test.cpu().detach().numpy().flatten()
        y_pred = mean.cpu().detach().numpy().flatten()
        y_err = ((upper - lower) / 2).cpu().detach().numpy().flatten()

        # Parity diagonal
        diag_min = y_true.min()
        diag_max = y_true.max()
        ax.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            kwargs.get('diag_ls', '-'),
            color=kwargs.get('diag_color', 'k'),
            linewidth=kwargs.get('diag_lw', 2),
            label=kwargs.get('diag_label', None),
        )

        ax.set_xlabel(kwargs.get('xlabel', 'Actual'))
        ax.set_ylabel(kwargs.get('ylabel', 'Predicted'))

        # Errorbar point style
        fmt = kwargs.get('fmt', '*')
        err_kwargs = kwargs.get('errorbar_kwargs', {}).copy()
        if 'color' not in err_kwargs and 'color' in kwargs:
            err_kwargs['color'] = kwargs['color']

        ax.errorbar(x=y_true, y=y_pred, yerr=y_err, fmt=fmt, **err_kwargs)

        # Metrics and title
        self.r2 = r2_score(y_true.reshape(-1), y_pred)
        title = kwargs.get('title', f"R2: {self.r2:.2f}, MSE: {test_error:.3f}")
        ax.set_title(title)

        return ax
    
    def getRange(self, X, y, threshold=0.05):
        """Get the range of input parameters corresponding to the top threshold percentile of output values.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of parameters.
        y : torch.Tensor
            Output tensor of values.
        threshold : float, optional
            Threshold percentile to select the top values, by default 0.05

        Returns
        -------
        torch.Tensor
            Tensor containing the new bounds for the input parameters.
        """        

        percentile_val = torch.quantile(y, threshold)
        mask = y <= percentile_val
        X_selected = X[mask.squeeze()]
        new_lb_mod = torch.min(X_selected, dim=0).values
        new_ub_mod = torch.max(X_selected, dim=0).values

        new_bounds = torch.stack((new_lb_mod, new_ub_mod), dim=0)
        self.new_bounds = new_bounds
        return new_bounds
    
    def calc_marg(self, n_points=10000, num_runs=5):
        """Calculate the marginal likelihood using Monte Carlo integration.

        This implementation is inspired by the work of Eunchi Kim, referenced in:
         
        Eunchi Kim et al., Solar RRL 2025, 9, e202500648 *“Inferring Material Parameters from Current-Voltage Curves in Organic Solar Cells via Neural Network-Based Surrogate Models”*, 

        and adapted from the original code:  

        https://github.com/eunchi-kim/Bayesian_inference_fourJV

        
        Parameters
        ----------
        n_points : int, optional
            Number of points to sample for Monte Carlo integration. Default is 10,000.
        num_runs : int, optional
            Number of Monte Carlo runs to average over. Default is 5.

        Returns
        -------
        float
            Estimated marginal likelihood.
        """
    

        if self.model is None:
            self.train_surrogate_model()

        
        lb_mod = self.bounds_tensor[0].detach().clone()
        ub_mod = self.bounds_tensor[1].detach().clone()
        

        if hasattr(self, 'new_bounds') and self.new_bounds is not None:
            new_lb_mod = self.new_bounds[0].detach().clone()
            new_ub_mod = self.new_bounds[1].detach().clone()
        else:
            self.new_bounds = self.getRange(self.X, self.y, threshold=0.05)
            new_lb_mod = self.new_bounds[0].detach().clone()
            new_ub_mod = self.new_bounds[1].detach().clone()
            
        vol = (torch.prod(new_ub_mod - new_lb_mod) / torch.prod(ub_mod - lb_mod)).item()

        mc_inte_rd = torch.zeros(num_runs, device=self.device)
        for run in range(num_runs):
            # Generate random parameters using torch 
            parmat = torch.rand(n_points, self.num_free_params, device=self.device) * (ub_mod - lb_mod) + lb_mod
            
            with torch.no_grad():  # Ensure no gradients are computed
                if self.log10_transform_LLH:
                    pred = self.model.posterior(parmat)
                    mean = pred.mean.detach()
                    mean = -torch.pow(10, mean)
                    # Use torch.logsumexp instead of scipy's logsumexp
                    y_eval = torch.logsumexp(mean.squeeze(), dim=0) - torch.log(torch.tensor(n_points, device=self.device)) + torch.log(torch.tensor(vol, device=self.device))
                else:
                    pred = self.model.posterior(parmat)
                    mean = pred.mean.detach()
                    y_eval = torch.logsumexp(mean.squeeze(), dim=0) - torch.log(torch.tensor(n_points, device=self.device)) + torch.log(torch.tensor(vol, device=self.device))
            
            mc_inte_rd[run] = y_eval.detach().item()
        
        marg_LH = torch.mean(torch.exp(mc_inte_rd)).item()
        
        self.marg_LH_log = torch_logsumexp(mc_inte_rd, dim=0).item() - torch.log(torch.tensor(num_runs, device=self.device)).item() 
        self.marg_LH = torch.exp(torch.tensor(self.marg_LH_log)).item()
        return marg_LH
        
    ##############################################################################################
    # Calculate 1D and 2D posterior slices
    ##############################################################################################
    # The following code is an adaptation of the code from Eunchi Kim et al., Solar RRL 2025, 9, e202500648
    # "Inferring Material Parameters from Current-Voltage Curves in Organic Solar Cells via Neural Network-Based Surrogate Models"
    # The main difference lies in the use of BoTorch GP models trained to learn the likelihood function instead of neural networks trained to learn the current-voltage curves. This allows for a more general approach to posterior estimation using surrogate models.
    def calculate_1d_posteriors_slice(self, param_name,Nres = 50, slice_params=None,slice_nat_scale=False,vmin=-100):
        """Calculate 1D posterior slice for a given parameter.
        If slice_params is None, use best_params as slice_params.

        Parameters
        ----------
        param_name : str
            Name of the parameter for which to calculate the 1D posterior slice.
        Nres : int, optional
            Number of resolution points for the slice, by default 50
        slice_params : dict, optional
            Dictionary of parameter values for the slice, by default None
        slice_nat_scale : bool, optional
            Whether to use natural scale for the slice parameters, by default False
        vmin : int, optional
            Minimum value for the posterior slice, by default -100

        Returns
        -------
        par_ax : np.ndarray
            Array of parameter values for the slice.
        LHS : np.ndarray
            Array of log-posterior values for the slice.
        vmin : int
            Minimum value for the posterior slice.
        vmax : int
            Maximum value for the posterior slice.
        """        
    
        # get marginal likelihood if not already done
        if not hasattr(self, 'marg_LH'):
            self.calc_marg(n_points=10000, num_runs=5)  
        
        if slice_params is None:
            slice_params = self.best_params
            if slice_nat_scale:
                slice_params = self.params_rescale(slice_params, self.params)

        # get parameter index
        param_idx = 0
        force_log = False
        for i, p in enumerate(self.params):
            if p.name != param_name:
                param_idx += 1
            else:
                # check is force_log is True
                if p.force_log:
                    force_log = True
                break

        par_ax = np.linspace(self.bounds_tensor[0][param_idx].cpu().numpy(), self.bounds_tensor[1][param_idx].cpu().numpy(), Nres)
        par_ax = np.sort(np.append(par_ax, slice_params[param_name]))
        #convert slice_params to list keep only values of free parameters
        list_slice_param = []
        for p in self.params:
            if not p.type == 'fixed':
                list_slice_param.append(slice_params[p.name])
        slice_params = np.array(list_slice_param)
        par_mat_mod = np.tile(slice_params, (Nres+1, 1))
        par_mat_mod[:, param_idx] = par_ax
        par_mat_tensor = torch.tensor(par_mat_mod).to(self.device)
        if self.log10_transform_LLH:
            pred = self.model.posterior(par_mat_tensor)
            mean = pred.mean
            mean = -torch.pow(10, mean)
            logLH = mean.cpu().detach().numpy().flatten()
        else:
            pred = self.model.posterior(par_mat_tensor)
            mean = pred.mean
            logLH = mean.cpu().detach().numpy().flatten()

        LHS = logLH - self.marg_LH_log#- np.log(self.marg_LH)
        LHS[LHS < vmin] = vmin
        vmax =int(np.max(LHS) + 1)

        par_ax = self.rescale_array(par_ax, self.params, param_name)

        return par_ax, LHS, vmin, vmax
        
    def calculate_2d_posteriors_slice(self, param_name_x, param_name_y, Nres = 50, slice_params=None,slice_nat_scale=False,vmin=-100):
        """Calculate a 2D posterior slice for two given parameters.

        This function computes the log-posterior values on a 2D grid for the specified parameters,
        while keeping all other parameters fixed (using `slice_params` or the best parameters by default).
        The posterior is normalized by the marginal likelihood.

        Parameters
        ----------
        param_name_x : str
            Name of the first parameter (x-axis) for the 2D posterior slice.
        param_name_y : str
            Name of the second parameter (y-axis) for the 2D posterior slice.
        Nres : int, optional
            Number of resolution points per axis for the slice (default: 50).
        slice_params : dict, optional
            Dictionary of parameter values to fix for all other parameters (default: None, uses best_params).
        slice_nat_scale : bool, optional
            If True, interpret `slice_params` in natural scale (default: False).
        vmin : int, optional
            Minimum value for the log-posterior (for clipping, default: -100).

        Returns
        -------
        par_ax_x : np.ndarray
            Array of x-axis parameter values (possibly rescaled to natural units).
        par_ax_y : np.ndarray
            Array of y-axis parameter values (possibly rescaled to natural units).
        LHS : np.ndarray
            2D array of log-posterior values for the parameter grid.
        vmin : int
            Minimum value used for clipping.
        vmax : int
            Maximum value of the log-posterior (rounded up).
        """        
        
        # get marginal likelihood if not already done
        if not hasattr(self, 'marg_LH'):
            self.calc_marg(n_points=10000, num_runs=5)  
        
        if slice_params is None:
            slice_params = self.best_params
            if slice_nat_scale:
                slice_params = self.params_rescale(slice_params, self.params)

        # get parameter indices
        param_idx_x = 0
        param_idx_y = 0
        got_x = False
        got_y = False
        for i, p in enumerate(self.params):
            if p.name != param_name_x and not got_x:
                param_idx_x += 1
            else:
                got_x = True
            if p.name != param_name_y and not got_y:
                param_idx_y += 1
            else:
                got_y = True
            if got_x and got_y:
                break
            
        par_ax_x = np.linspace(self.bounds_tensor[0][param_idx_x].cpu().numpy(), self.bounds_tensor[1][param_idx_x].cpu().numpy(), Nres)
        par_ax_y = np.linspace(self.bounds_tensor[0][param_idx_y].cpu().numpy(), self.bounds_tensor[1][param_idx_y].cpu().numpy(), Nres)
        #convert slice_param to list keep only values of free parameters
        list_slice_param = []
        for p in self.params:
            if not p.type == 'fixed':
                list_slice_param.append(slice_params[p.name])
        # append the slice values to the axes
        par_ax_x = np.sort(np.append(par_ax_x, slice_params[param_name_x]))
        par_ax_y = np.sort(np.append(par_ax_y, slice_params[param_name_y]))

        slice_params = np.array(list_slice_param)
        par_mat_mod = np.tile(slice_params, ((Nres + 1) ** 2, 1))
        for idx, (val_x, val_y) in enumerate(itertools.product(par_ax_x, par_ax_y)):
            par_mat_mod[idx, param_idx_x] = val_x
            par_mat_mod[idx, param_idx_y] = val_y

        par_mat_tensor = torch.tensor(par_mat_mod).to(self.device)

        if self.log10_transform_LLH:
            pred = self.model.posterior(par_mat_tensor)
            mean = pred.mean
            mean = -torch.pow(10, mean)
            logLH = mean.cpu().detach().numpy().flatten()
        else:
            pred = self.model.posterior(par_mat_tensor)
            mean = pred.mean
            logLH = mean.cpu().detach().numpy().flatten()
        LHS = logLH - self.marg_LH_log#- np.log(self.marg_LH)
        LHS[LHS < vmin] = vmin
        vmax =int(np.max(LHS) + 1)

        par_ax_x = self.rescale_array(par_ax_x, self.params, param_name_x)
        par_ax_y = self.rescale_array(par_ax_y, self.params, param_name_y)

        return par_ax_x, par_ax_y, LHS.reshape(Nres+1, Nres+1), vmin, vmax

    def plot_1d_posteriors_slice(self, param_name, Nres=50, slice_params=None, slice_nat_scale=False, vmin=-100, ax=None, **kwargs):
        """Plot the 1D posterior slice for a given parameter.

        This function visualizes the log-posterior values along a single parameter axis,
        keeping all other parameters fixed (using `slice_params` or the best parameters by default).
        The posterior is normalized by the marginal likelihood.

        Parameters
        ----------
        param_name : str
            Name of the parameter for which to plot the 1D posterior slice.
        Nres : int, optional
            Number of resolution points for the slice (default: 50).
        slice_params : dict, optional
            Dictionary of parameter values to fix for all other parameters (default: None, uses best_params).
        slice_nat_scale : bool, optional
            If True, interpret `slice_params` in natural scale (default: False).
        vmin : int, optional
            Minimum value for the log-posterior (for clipping, default: -100).
        ax : matplotlib.axes.Axes or None, optional
            Existing axes to draw into. If None, a new figure/axes are created internally.
        **kwargs : dict, optional
            Additional keyword arguments for plot customization (e.g., color, figsize, label, linestyle).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated matplotlib figure.
        ax : matplotlib.axes.Axes
            The axes of the figure.
        """
        par_ax, LHS, vmin, vmax = self.calculate_1d_posteriors_slice(param_name, Nres, slice_params, slice_nat_scale, vmin)

        for p in self.params:
            if p.name == param_name:
                xlabel = p.display_name + ' [' + p.unit + ']'

        # Create axes if not provided
        if ax is None:
            _ , ax = plt.subplots(figsize=kwargs.get('figsize', (6, 4)))
        
        ax.plot(par_ax, LHS, color=kwargs.get('color', 'C0'), label=kwargs.get('label', None), linestyle=kwargs.get('linestyle', '-'))
        if kwargs.get('show_best_param', True):
            ax.axvline(self.true_best_params[param_name], color=kwargs.get('best_param_color', 'red'), linestyle=kwargs.get('best_param_ls', '--'))

        if param_name in self.take_log_params_names:
            ax.set_xscale('log')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(kwargs.get('ylabel', 'Log Posterior'))
        ax.set_ylim(vmin, vmax)
        if kwargs.get('legend', False):
            ax.legend()
        return ax
    
    def plot_2d_posteriors_slice(self, param_name_x, param_name_y, Nres=50, slice_params=None, slice_nat_scale=False, vmin=-100, ax=None, **kwargs):
        """
        Plot the 2D posterior slice for two given parameters.

        Parameters
        ----------
        param_name_x : str
            Name of the x-axis parameter.
        param_name_y : str
            Name of the y-axis parameter.
        Nres : int, optional
            Number of resolution points per axis (default: 50).
        slice_params : dict, optional
            Parameter values to fix for all other parameters (default: None, uses best_params).
        slice_nat_scale : bool, optional
            If True, interpret `slice_params` in natural scale (default: False).
        vmin : int, optional
            Minimum value for the log-posterior (for clipping, default: -100).
        ax : matplotlib.axes.Axes or None, optional
            Existing axes to draw into. If None, a new figure/axes are created.
        **kwargs : dict, optional
            Additional keyword arguments for plot customization (e.g., cmap, markersize, figsize).

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes of the figure.
        """
        par_ax_x, par_ax_y, LHS, vmin, vmax = self.calculate_2d_posteriors_slice(
            param_name_x, param_name_y, Nres, slice_params, slice_nat_scale, vmin
        )

        # Get axis labels
        xlabel, ylabel = "", ""
        for p in self.params:
            if p.name == param_name_x:
                xlabel = p.display_name + ' [' + p.unit + ']'
            if p.name == param_name_y:
                ylabel = p.display_name + ' [' + p.unit + ']'

        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get('figsize', (6, 5)))

        # Plot the contour
        c = ax.contourf(
            par_ax_x, par_ax_y, LHS.T,
            levels=kwargs.get('levels', 50),
            vmin=vmin, vmax=vmax,
            cmap=kwargs.get('cmap', 'viridis')
        )

        # Best parameter markers/lines
        if kwargs.get('show_best_param', True):
            ax.axvline(self.true_best_params[param_name_x], color=kwargs.get('best_param_color', 'red'), linestyle=kwargs.get('best_param_ls', '--'))
            ax.axhline(self.true_best_params[param_name_y], color=kwargs.get('best_param_color', 'red'), linestyle=kwargs.get('best_param_ls', '--'))
            ax.plot(
                self.true_best_params[param_name_x],
                self.true_best_params[param_name_y],
                kwargs.get('best_param_marker', 'r*'),
                markersize=kwargs.get('markersize', 12)
            )

        # Log scales if needed
        if param_name_x in self.take_log_params_names:
            ax.set_xscale('log')
        if param_name_y in self.take_log_params_names:
            ax.set_yscale('log')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(par_ax_x.min(), par_ax_x.max())
        ax.set_ylim(par_ax_y.min(), par_ax_y.max())

        # Optionally add colorbar
        if kwargs.get('colorbar', False):
            plt.colorbar(c, ax=ax)

        return ax
    
    def plot_all_posteriors_slices(self, Nres = 50, slice_params=None, slice_nat_scale=False, vmin=-100, **kwargs):
        """ Plot all 1D and 2D posterior slices for the parameters in a grid layout.

        Parameters
        ----------
        Nres : int, optional
            Number of resolution points per axis (default: 50).
        slice_params : dict, optional
            Parameter values to fix for all other parameters (default: None, uses best_params).
        slice_nat_scale : bool, optional
            If True, interpret `slice_params` in natural scale (default: False).
        vmin : int, optional
            Minimum value for the log-posterior (for clipping, default: -100).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated matplotlib figure.
        axes : np.ndarray
            The axes array of the figure.
        
        """    
        num_params = self.num_free_params
        fig, axes = plt.subplots(num_params, num_params, figsize=kwargs.get('figsize', (4*num_params, 4*num_params)))
        param_names = [p.name for p in self.params if not p.type == 'fixed']

        # build quick lookup from name -> param object for bounds/display
        param_by_name = {p.name: p for p in self.params}

        for row_idx, col_idx in itertools.product(range(num_params), range(num_params)):
            x_name = param_names[col_idx]  # column -> x-axis param
            y_name = param_names[row_idx]  # row -> y-axis param
            ax = axes[row_idx, col_idx]

            # Diagonal: 1D slice for the parameter (tied to both x and y same param)
            if row_idx == col_idx:
                par_ax, LHS, vmin_slice, vmax_slice = self.calculate_1d_posteriors_slice(
                    x_name, Nres, slice_params, slice_nat_scale, vmin
                )
                ax.plot(par_ax, LHS, color=kwargs.get('color', 'C0'))
                ax.axvline(self.true_best_params[x_name], color='red', linestyle='--')

                if x_name in self.take_log_params_names:
                    ax.set_xscale('log')

                # x-limits from parameter bounds
                p = param_by_name[x_name]
                ax.set_xlim(p.bounds[0], p.bounds[1])
                ax.set_ylim(vmin, vmax_slice)

                # y-labels: left column shows ylabel, other columns show right ticks/empty
                if col_idx == 0:
                    ax.set_ylabel(kwargs.get('ylabel_diag','Log Posterior Slice'))
                else:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('')

                # x-label only on bottom row
                if row_idx == num_params - 1:
                    ax.set_xlabel(f"{p.display_name} [{p.unit}]")
                else:
                    ax.set_xticklabels([])

            # Off-diagonal: 2D slice, x = column param, y = row param
            elif row_idx > col_idx:
                par_ax_x, par_ax_y, LHS, vmin_slice, vmax_slice = self.calculate_2d_posteriors_slice(
                    x_name, y_name, Nres, slice_params, slice_nat_scale, vmin
                )

                # LHS orientation: keep same as your original code (transpose if needed)
                c = ax.contourf(
                    par_ax_x, par_ax_y, LHS.T,
                    levels=50, vmin=vmin, vmax=vmax_slice,
                    cmap=kwargs.get('cmap', 'viridis')
                )

                # best params markers/lines
                ax.axvline(self.true_best_params[x_name], color='red', linestyle='--')
                ax.axhline(self.true_best_params[y_name], color='red', linestyle='--')
                ax.plot(
                    self.true_best_params[x_name],
                    self.true_best_params[y_name],
                    'r*', markersize=kwargs.get('markersize', 12)
                )

                # set axis bounds from parameter definitions
                px = param_by_name[x_name]
                py = param_by_name[y_name]
                ax.set_xlim(px.bounds[0], px.bounds[1])
                ax.set_ylim(py.bounds[0], py.bounds[1])

                # log scales if requested
                if x_name in self.take_log_params_names:
                    ax.set_xscale('log')
                if y_name in self.take_log_params_names:
                    ax.set_yscale('log')

                # x-label only for bottom row (same x label for that column)
                if row_idx == num_params - 1:
                    ax.set_xlabel(f"{px.display_name} [{px.unit}]")
                else:
                    ax.set_xticklabels([])

                # y-label only for first column (same y label for that row)
                if col_idx == 0:
                    ax.set_ylabel(f"{py.display_name} [{py.unit}]")
                else:
                    ax.set_yticklabels([])
            else:
                ax.axis('off')

        plt.tight_layout()
        return fig, axes
    
    ##############################################################################################
    # Calculate 1D and 2D posterior with full grid
    ##############################################################################################
    # In the following we really calculate the marginalized posterior from the full grid of posterior values.
    # This is different from the slicing approach above, where we fix other parameters to specific values.
    ##############################################################################################

    def calculate_posterior_grid(self, Nres = 5, vmin=-100):
        """Calculate the full grid of posterior values for all parameter combinations.

        Parameters
        ----------
        Nres : int, optional
            Number of resolution points per parameter, by default 50
        vmin : float, optional
            Minimum value for the posterior, by default -100
        slice_param : str or None, optional
            Parameter to slice on, by default None
        slice_nat_scale : bool, optional
            Whether to slice in natural scale, by default False

        Returns
        -------
        param_grids : list of np.ndarray
            List of parameter grids for each parameter.
        posterior_grid : np.ndarray
            Grid of posterior values.
        vmin : float
            Minimum value used for clipping.
        vmax : float
            Maximum value used for clipping.
        """
        # get marginal likelihood if not already done
        if not hasattr(self, 'marg_LH'):
            self.calc_marg(n_points=10000, num_runs=5)  

        param_axes = []
        param_names = []
        for i, p in enumerate(self.params):
            if not p.type == 'fixed':
                param_names.append(p.name)
                par_ax = np.linspace(self.bounds_tensor[0][i].cpu().numpy(), self.bounds_tensor[1][i].cpu().numpy(), Nres)
                # append best param value to axis
                par_ax = np.sort(np.append(par_ax, self.best_params[p.name]))
                param_axes.append(par_ax)

        # create meshgrid
        mesh = np.meshgrid(*param_axes, indexing='ij')
        grid_shape = mesh[0].shape
        grid_points = np.vstack([m.flatten() for m in mesh]).T

        par_mat_tensor = torch.tensor(grid_points).to(self.device)

        if self.log10_transform_LLH:
            pred = self.model.posterior(par_mat_tensor)
            mean = pred.mean
            mean = -torch.pow(10, mean)
            logLH = mean.cpu().detach().numpy().flatten()
        else:
            pred = self.model.posterior(par_mat_tensor)
            mean = pred.mean
            logLH = mean.cpu().detach().numpy().flatten()
        
        LHS = logLH - self.marg_LH_log#- np.log(self.marg_LH)
        LHS[LHS < vmin] = vmin
        vmax =int(np.max(LHS) + 1)

        posterior_grid = LHS.reshape(grid_shape)

        # rescale parameter axes to natural scale
        for i, p in enumerate(self.params):
            if not p.type == 'fixed':
                param_axes[i] = self.rescale_array(param_axes[i], self.params, p.name)

        return param_axes, posterior_grid, vmin, vmax
    
    def calculate_1d_posteriors_marginal_grid(self, param_name, Nres = 5, vmin=-100):
        """Calculate 1D posterior for a given parameter from the full grid.

        Parameters
        ----------
        param_name : str
            Name of the parameter to calculate the posterior for.
        Nres : int, optional
            Number of resolution points, by default 50
        vmin : float, optional
            Minimum value for clipping, by default -100

        Returns
        -------
        par_ax : np.ndarray
            Parameter axis values.
        LHS : np.ndarray
            Log posterior values.
        vmin : float
            Minimum value used for clipping.
        vmax : float
            Maximum value used for clipping.
        """
        param_axes, posterior_grid, vmin, vmax = self.calculate_posterior_grid(Nres, vmin)

        # get parameter index
        param_idx = 0
        for i, p in enumerate(self.params):
            if p.name != param_name:
                param_idx += 1
            else:
                break

        # marginalize over other parameters
        LHS = logsumexp(posterior_grid, axis=tuple(i for i in range(len(param_axes)) if i != param_idx))

        par_ax = param_axes[param_idx]

        return par_ax, LHS, vmin, vmax
    
    def calculate_2d_posteriors_marginal_grid(self, param_name_x, param_name_y, Nres = 5, vmin=-100):
        """Calculate 2D posterior for given parameters from the full grid.

        Parameters
        ----------
        param_name_x : str
            Name of the x-axis parameter.
        param_name_y : str
            Name of the y-axis parameter.
        Nres : int, optional
            Number of resolution points, by default 50
        vmin : float, optional
            Minimum value for clipping, by default -100

        Returns
        -------
        par_ax_x : np.ndarray
            X-axis parameter values.
        par_ax_y : np.ndarray
            Y-axis parameter values.
        LHS : np.ndarray
            Log posterior values.
        vmin : float
            Minimum value used for clipping.
        vmax : float
            Maximum value used for clipping.
        """
        param_axes, posterior_grid, vmin, vmax = self.calculate_posterior_grid(Nres, vmin)

        # get parameter indices
        param_idx_x = 0
        param_idx_y = 0
        got_x = False
        got_y = False
        for i, p in enumerate(self.params):
            if p.name != param_name_x and not got_x:
                param_idx_x += 1
            else:
                got_x = True
            if p.name != param_name_y and not got_y:
                param_idx_y += 1
            else:
                got_y = True
            if got_x and got_y:
                break

        # marginalize over other parameters
        LHS = logsumexp(posterior_grid, axis=tuple(i for i in range(len(param_axes)) if i != param_idx_x and i != param_idx_y))

        par_ax_x = param_axes[param_idx_x]
        par_ax_y = param_axes[param_idx_y]

        return par_ax_x, par_ax_y, LHS, vmin, vmax

    def plot_1d_posteriors_marginal_grid(self, param_name, Nres = 5, vmin=-100, ax=None, **kwargs):
        """Plot 1D posterior for a given parameter from the full grid.

        Parameters
        ----------
        param_name : str
            Name of the parameter to plot.
        Nres : int, optional
            Number of resolution points, by default 5
        vmin : float, optional
            Minimum value for clipping, by default -100
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw into. If None, a new figure/axes are created internally.
        **kwargs : dict, optional
            Additional keyword arguments for plot customization (e.g., color, figsize).

        Returns
        -------

        ax : matplotlib.axes.Axes
            The axes of the figure.
        """
        par_ax, LHS, vmin, vmax = self.calculate_1d_posteriors_marginal_grid(param_name, Nres, vmin)

        for p in self.params:
            if p.name == param_name:
                xlabel = p.display_name + ' [' +p.unit+']'
        # Create axes if not provided
        if ax is None:
            _ , ax = plt.subplots(figsize=kwargs.get('figsize', (6, 4)))

        sc = ax.plot(par_ax, LHS, color=kwargs.get('color', 'C0'))
        ax.axvline(self.true_best_params[param_name], color='red', linestyle='--')
        
        if param_name in self.take_log_params_names:
            ax.set_xscale('log')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(kwargs.get('ylabel', 'Log Posterior'))
        ax.set_ylim(vmin, vmax)
        return ax
    
    def plot_2d_posteriors_marginal_grid(self, param_name_x, param_name_y, Nres = 5, vmin=-100, ax=None, **kwargs):
        """Plot 2D posterior for given parameters from the full grid.

        Parameters
        ----------
        param_name_x : str
            Name of the x-axis parameter.
        param_name_y : str
            Name of the y-axis parameter.
        Nres : int, optional
            Number of resolution points, by default 5
        vmin : float, optional
            Minimum value for clipping, by default -100
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw into. If None, a new figure/axes are created internally.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes of the figure.
        """
        par_ax_x, par_ax_y, LHS, vmin, vmax = self.calculate_2d_posteriors_marginal_grid(param_name_x, param_name_y, Nres, vmin)

        for p in self.params:
            if p.name == param_name_x:
                xlabel = p.display_name + ' [' +p.unit+']'
            if p.name == param_name_y:
                ylabel = p.display_name + ' [' +p.unit+']'


        if ax is None:
            _ , ax = plt.subplots(figsize=kwargs.get('figsize', (6,5)))
        c = ax.contourf(par_ax_x, par_ax_y, LHS.T, levels=50, vmin=vmin, vmax=vmax, cmap=kwargs.get('cmap', 'viridis'))
        ax.axvline(self.true_best_params[param_name_x], color='red', linestyle='--')
        ax.axhline(self.true_best_params[param_name_y], color='red', linestyle='--')
        ax.plot(self.true_best_params[param_name_x], self.true_best_params[param_name_y], 'r*', markersize=kwargs.get('markersize', 12))
        if param_name_x in self.take_log_params_names:
            ax.set_xscale('log')
        if param_name_y in self.take_log_params_names:
            ax.set_yscale('log')    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax
    
    def plot_all_posteriors_marginal_grid(self, Nres = 5, vmin=-100, **kwargs):
        """ Plot all 1D and 2D posterior slices from the full grid in a grid layout.

        Parameters
        ----------
        Nres : int, optional
            Number of resolution points, by default 5
        vmin : int, optional
            Minimum value for clipping, by default -100

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated matplotlib figure.
        axes : np.ndarray
            The axes array of the figure.
        """        
        num_params = self.num_free_params
        fig, axes = plt.subplots(num_params, num_params, figsize=kwargs.get('figsize', (4*num_params, 4*num_params)))
        param_names = [p.name for p in self.params if not p.type == 'fixed']

        # build quick lookup from name -> param object for bounds/display
        param_by_name = {p.name: p for p in self.params}

        # Calculate posterior grid once
        param_axes, posterior_grid, vmin_slice, vmax_slice = self.calculate_posterior_grid(Nres, vmin)

        for row_idx, col_idx in itertools.product(range(num_params), range(num_params)):
            x_name = param_names[col_idx]  # column -> x-axis param
            y_name = param_names[row_idx]  # row -> y-axis param
            ax = axes[row_idx, col_idx]

            # Diagonal: 1D slice for the parameter (tied to both x and y same param)
            if row_idx == col_idx:
                # Marginalize over other parameters
                LHS = logsumexp(posterior_grid, axis=tuple(i for i in range(num_params) if i != col_idx))
                par_ax = param_axes[col_idx]
                
                ax.plot(par_ax, LHS, color=kwargs.get('color', 'C0'))
                ax.axvline(self.true_best_params[x_name], color='red', linestyle='--')

                if x_name in self.take_log_params_names:
                    ax.set_xscale('log')

                # x-limits from parameter bounds
                p = param_by_name[x_name]
                ax.set_xlim(p.bounds[0], p.bounds[1])
                ax.set_ylim(vmin, vmax_slice)

                # y-labels: left column shows ylabel, other columns show right ticks/empty
                if col_idx == 0:
                    ax.set_ylabel('Log Posterior Slice')
                else:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('')
                # x-label only on bottom row
                if row_idx == num_params - 1:
                    ax.set_xlabel(f"{p.display_name} [{p.unit}]")
                else:
                    ax.set_xticklabels([])
            # Off-diagonal: 2D slice, x = column param, y = row param
            elif row_idx > col_idx:
                # Marginalize over other parameters
                LHS = logsumexp(posterior_grid, axis=tuple(i for i in range(num_params) if i != col_idx and i != row_idx))
                par_ax_x = param_axes[col_idx]
                par_ax_y = param_axes[row_idx]

                # LHS orientation: keep same as your original code (transpose if needed)
                c = ax.contourf(
                    par_ax_x, par_ax_y, LHS.T,
                    levels=kwargs.get('levels',50), 
                    vmin=vmin, vmax=vmax_slice,
                    cmap=kwargs.get('cmap', 'viridis')
                )

                # best params markers/lines
                ax.axvline(self.true_best_params[x_name], color='red', linestyle='--')
                ax.axhline(self.true_best_params[y_name], color='red', linestyle='--')
                ax.plot(
                    self.true_best_params[x_name],
                    self.true_best_params[y_name],
                    'r*', markersize=kwargs.get('markersize', 12)
                )

                # set axis bounds from parameter definitions
                px = param_by_name[x_name]
                py = param_by_name[y_name]
                ax.set_xlim(px.bounds[0], px.bounds[1])
                ax.set_ylim(py.bounds[0], py.bounds[1])

                # log scales if requested
                if x_name in self.take_log_params_names:
                    ax.set_xscale('log')
                if y_name in self.take_log_params_names:
                    ax.set_yscale('log')

                # x-label only for bottom row (same x label for that column)
                if row_idx == num_params - 1:
                    ax.set_xlabel(f"{px.display_name} [{px.unit}]")
                else:
                    ax.set_xticklabels([])

                # y-label only for first column (same y label for that row)
                if col_idx == 0:
                    ax.set_ylabel(f"{py.display_name} [{py.unit}]")
                else:
                    ax.set_yticklabels([])
            else:
                ax.axis('off')

        # clear cuda memory cache
        # torch.cuda.empty_cache()
        plt.tight_layout()
        return fig, axes
    

    ##############################################################################################
    # Calculate 1D and 2D posterior with pyro MCMC NUTS
    ##############################################################################################
    # In the following we use MCMC NUTS to sample from the posterior using the surrogate model.
    ##############################################################################################
    
    # def calculate_posteriors_mcmc_nuts(self, num_samples=1000, warmup_steps=200, num_chains=1, vmin=-100):
    #     """Calculate the posterior using Pyro MCMC NUTS.

    #     Parameters
    #     ----------
    #     num_samples : int, optional
    #         Number of samples to draw, by default 1000
    #     warmup_steps : int, optional
    #         Number of warmup steps, by default 200
    #     num_chains : int, optional
    #         Number of chains to run, by default 1
    #     vmin : float, optional
    #         Minimum value for clipping, by default -100

    #     Returns
    #     -------
    #     param_samples : dict
    #         Dictionary of parameter samples.
    #     log_posterior_samples : np.ndarray
    #         Array of log posterior samples.
    #     vmin : float
    #         Minimum value used for clipping.
    #     vmax : float
    #         Maximum value used for clipping.
    #     """
    #     # if self.model is None:
    #         # self.train_surrogate_model()

    #     if not hasattr(self, 'marg_LH'):
    #         self.calc_marg(n_points=10000, num_runs=5)  
    #     # print(type(self.marg_LH), self.marg_LH)
    #     # Set up NUTS sampler
    #     nuts_kernel = NUTS(self.pyro_model)
    #     mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, mp_context='spawn', disable_progbar=True)
    #     mcmc.run(torch.log(torch.tensor(self.marg_LH).to(self.device)))
    #     samples = mcmc.get_samples()
    #      #plot with arvi traceplot
    #     az_data = az.from_pyro(mcmc)
    #     az.plot_trace(az_data)
    #     az.plot_autocorr(az_data)
    #     az.plot_posterior(az_data)
    #     az.plot_pair(az_data, kind='kde', marginals=True)
    #     plt.show()

    def calculate_posteriors_mcmc_nuts(self, num_samples=1000, warmup_steps=200, num_chains=1, **kwargs):
        """Calculate the posterior using Pyro MCMC NUTS.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw, by default 1000
        warmup_steps : int, optional
            Number of warmup steps, by default 200
        num_chains : int, optional
            Number of chains to run, by default 1
        vmin : float, optional
            Minimum value for clipping, by default -100

        Returns
        -------
        param_samples : dict
            Dictionary of parameter samples.
        log_posterior_samples : np.ndarray
            Array of log posterior samples.
        vmin : float
            Minimum value used for clipping.
        vmax : float
            Maximum value used for clipping.
        """
        # if self.model is None:
            # self.train_surrogate_model()

        parallel_chains = kwargs.get('parallel_chains', True)
        if num_chains > 1 and self.device == 'cuda' and parallel_chains:
            logger.warning("Multiple chains with CUDA device may lead to unexpected behavior. We will run the chains sequentially instead.")
            parallel_chains = False

        initialize_with_best = kwargs.get('initialize_with_best', False)

        if initialize_with_best: # get the best len(num_chains) parameters from self.y and self.X
            sorted_indices = torch.argsort(self.y,dim=0) # get indices
            # get best parameters
            best_params_init= self.X[sorted_indices[-num_chains:]].to(self.device)
            initial_params = {}
            for idx, name in enumerate(self.name_free_params):
                

                initial_params[name] = best_params_init.squeeze()[:,idx]
            # print("Initializing MCMC chains with best parameters:", initial_params)


            
        if not hasattr(self, 'marg_LH'):
            self.calc_marg(n_points=10000, num_runs=5)  
        # print(type(self.marg_LH), self.marg_LH)
        # Set up NUTS sampler
        
        
        nuts_kernel = NUTS(self.pyro_model)
        # nuts_kernel = HMC(self.pyro_model,step_size=0.01)
        # nuts_kernel = RandomWalkKernel(self.pyro_model)
        if not parallel_chains:
            all_samples = []
            for chain in range(num_chains):
                logger.info(f"Running chain {chain+1}/{num_chains}...")
                if initialize_with_best:
                    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1, mp_context='spawn', disable_progbar=False, initial_params=initial_params[chain])
                else:
                    nuts_kernel = NUTS(self.pyro_model)
                    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1, mp_context='spawn', disable_progbar=False)
                mcmc.run(torch.log(torch.tensor(self.marg_LH).to(self.device)))
                samples = mcmc.get_samples()
                all_samples.append(samples)
            # Combine samples from all chains
            combined_samples = {}
            for key in all_samples[0].keys():
                combined_samples[key] = torch.cat([chain_samples[key] for chain_samples in all_samples], dim=0)
            samples = combined_samples
        else:
            if initialize_with_best:
                mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, mp_context='spawn', disable_progbar=True, initial_params=initial_params)
            else:
                mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, mp_context='spawn', disable_progbar=True)
            mcmc.run(torch.log(torch.tensor(self.marg_LH).to(self.device)))
            samples = mcmc.get_samples()
         #plot with arvi traceplot
        
        # try:
        #     az_data = az.convert_to_inference_data(az_data)
        # except Exception as e:
        az_data = az.from_pyro(mcmc)
        # az.plot_trace(az_data)
        # az.plot_autocorr(az_data)
        # az.plot_posterior(az_data, point_estimate='median')
        # az.plot_pair(az_data, kind='kde', marginals=True)
        # plt.show()

        return mcmc, samples, az_data

    # Define the model for Pyro MCMC
    def pyro_model(self,marg_LH=None):
        theta = []
        for idx, name in enumerate(self.name_free_params):
            # loc = (self.bounds_tensor[0][idx] + self.bounds_tensor[1][idx])/2
            # scale = (self.bounds_tensor[1][idx] - self.bounds_tensor[0][idx])/4
            # theta.append(pyro.sample(name, dist.Normal(loc, scale).to_event(0)))
            theta.append(pyro.sample(name, dist.Uniform(1.05*self.bounds_tensor[0][idx], 0.95*self.bounds_tensor[1][idx])))
        theta = torch.stack(theta).to(self.device)
        if self.log10_transform_LLH:
            pred = self.model.posterior(theta.unsqueeze(0))
            mean = pred.mean
            logp = -torch.pow(10, mean) - marg_LH
        else:
            pred = self.model.posterior(theta.unsqueeze(0))
            logp = pred.mean - marg_LH

        pyro.factor("logp", logp)
    
if __name__ == "__main__":
    pass
    

