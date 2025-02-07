import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy.special import logsumexp
from itertools import combinations
import ax
from ax import *
from ax.core.observation import ObservationFeatures
from ax.core.base_trial import TrialStatus as T
from optimpv.general.general import inv_loss_function



def get_MSE_grid(params, Nres, objective_name, model, loss, optimizer_type = 'ax'):
    """ Calculate the Mean Squared Error (MSE) grid for the given parameters and model.

    Parameters
    ----------
    params : list of FitParam() objects
        List of parameters to explore.
    Nres : int
        Number of points to evaluate for each parameter.
    objective_name : str
        Name of the objective to evaluate.
    model : torch model
        Model to evaluate the objective.
    loss : str
        Loss function used, see optimpv/general/general.py for available loss functions.
    optimizer_type : str, optional
        Type of optimizer used, by default 'ax'

    Returns
    -------
    np.ndarray
        Grid of MSE values.

    Raises
    ------
    ValueError
        If the optimizer type is not recognized.
    """    
    dims_GP, dims = [], []
    dic_fixed = {}
    for idx, param in enumerate(params):
        if param.type != 'fixed':
            if param.axis_type == 'log':
                if param.force_log:
                    parax = np.linspace(np.log10(param.bounds[0]),np.log10(param.bounds[1]),Nres)
                else:
                    parax = np.logspace(np.log10(param.bounds[0]/param.fscale),np.log10(param.bounds[1]/param.fscale),Nres)
                parax_rescaled = np.logspace(np.log10(param.bounds[0]),np.log10(param.bounds[1]),Nres)
            else:
                parax = np.linspace(param.bounds[0]/param.fscale,param.bounds[1]/param.fscale,Nres)
                parax_rescaled = np.linspace(param.bounds[0],param.bounds[1],Nres)
            dims_GP.append(parax)
            dims.append(parax_rescaled)
        else:
            dic_fixed[param.name] = param.value
    Xc = np.array(list(itertools.product(*dims_GP)))
    mean_predictions = np.zeros(len(Xc))
    observation_features = []
    if optimizer_type.lower() == 'ax':
        for i,line in enumerate(Xc):
            dum_dic = {}
            for idx, param in enumerate(params):
                dum_dic[param.name] = line[idx]
            dum_dic.update(dic_fixed)
            observation_features.append(ObservationFeatures(parameters=dum_dic))

        predictions = model.predict(observation_features)
        mean_predictions = np.array(predictions[1][objective_name][objective_name])
    else:
        raise ValueError('Optimizer type not recognized')
    
    # invert the loss
    mean_predictions = inv_loss_function(mean_predictions, loss) 

    return mean_predictions.reshape(*[Nres for i in range(len(params))]) , dims_GP, dims

# grid_MSE, dims_GP = get_MSE_grid(params, Nres, objective_name, model, loss, optimizer_type = 'ax')


def calculate_1d_posteriors(mse_array):
    """Calculate 1D posterior distributions over each parameter axis from an n x n-dimensional MSE array.

    Parameters
    ----------
    mse_array : np.ndarray
        An n-dimensional array of MSE values.

    Returns
    -------
    list
        A list of 1D posterior distributions for each parameter.
    """    
    # Convert MSE to negative log-likelihood
    negative_log_likelihood = -0.5 * mse_array

    # Compute the log posterior by normalizing using logsumexp
    log_posterior = negative_log_likelihood - logsumexp(negative_log_likelihood)

    # Calculate marginal posteriors for each parameter by summing over all other axes
    marginal_posteriors = []
    for axis in range(mse_array.ndim):
        # Use logsumexp to marginalize efficiently
        marginal_log = logsumexp(log_posterior, axis=tuple(i for i in range(mse_array.ndim) if i != axis))
        marginal_posteriors.append(np.exp(marginal_log))

    return marginal_posteriors

def calculate_2d_posteriors(mse_array):
    """Calculate 2D posterior distributions over each pair of parameter axes from an n x n-dimensional MSE array.

    Parameters
    ----------
    mse_array : np.ndarray
        An n-dimensional array of MSE values.

    Returns
    -------
    list
        A list of 2D posterior distributions for each pair of parameters.
    """    
    # Convert MSE to negative log-likelihood
    negative_log_likelihood = -0.5 * mse_array

    # Compute the log posterior by normalizing using logsumexp
    log_posterior = negative_log_likelihood - logsumexp(negative_log_likelihood)

    # Calculate pairwise marginal posteriors for each pair of parameters
    pairwise_posteriors = []
    ndim = mse_array.ndim
    for axis1 in range(ndim):
        for axis2 in range(axis1 + 1, ndim):
            # Marginalize over all other axes except axis1 and axis2
            marginal_log = logsumexp(log_posterior, axis=tuple(i for i in range(ndim) if i != axis1 and i != axis2))
            pairwise_posteriors.append(np.exp(marginal_log))

    return pairwise_posteriors

def devils_plot(params, Nres, objective_name, model, loss, best_parameters = None, params_orig = None, grid_MSE = None, dims_GP = None, optimizer_type = 'ax', **kwargs):
    """Generate a devil's plot to visualize the posterior distributions of parameters.

    Parameters
    ----------
    params : list of FitParam() objects
        List of parameters to explore.
    Nres : int
        Number of points to evaluate for each parameter.
    objective_name : str
        Name of the objective to evaluate.
    model : torch model
        Model to evaluate the objective.
    loss : str
        Loss function used, see optimpv/general/general.py for available loss functions.
    best_parameters : dict, optional
        Dictionary of the best parameters, by default None.
    params_orig : dict, optional
        Dictionary of the original parameters, by default None.
    optimizer_type : str, optional
        Type of optimizer used, by default 'ax'.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """    

    fig_size = kwargs.get('fig_size', (15, 15))
    marker_size = kwargs.get('marker_size', 200)
    if grid_MSE is None or dims_GP is None:
        grid_MSE, dims_GP, dims = get_MSE_grid(params, Nres, objective_name, model, loss, optimizer_type = optimizer_type)
    marginal_posteriors = calculate_1d_posteriors(grid_MSE)
    pairwise_posteriors = calculate_2d_posteriors(grid_MSE)

    n = len(params)
    names = [ param.name for param in params if param.type != 'fixed']
    comb = list(itertools.combinations(names, 2))

    dims_GP = dims
    fig, ax = plt.subplots(n, n, figsize=fig_size)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax[i, j].plot(dims_GP[j], marginal_posteriors[j])
                if params_orig is not None:
                    ax[i, j].axvline(x=params_orig[params[j].name], color='k', linestyle='-')
                if best_parameters is not None:
                    ax[i, j].axvline(x=best_parameters[params[j].name], color='tab:red', linestyle='--')
                if params[j].axis_type == 'log':
                    ax[i, j].set_xscale('log')
                ax[i, j].set_xlabel(params[j].display_name + ' [' +params[j].unit+']')
                ax[i, j].set_ylabel("Posterior probability")
                
            elif i > j:
                ax[i, j].contourf(dims_GP[j], dims_GP[i], pairwise_posteriors[comb.index((params[j].name, params[i].name))].reshape(Nres, Nres).T)
                if params_orig is not None:
                    ax[i,j].axhline(y=params_orig[params[i].name], color='k', linestyle='-')
                    ax[i,j].axvline(x=params_orig[params[j].name], color='k', linestyle='-')
                    # ax[i, j].scatter(params_orig[params[j].name], params_orig[params[i].name], c='tab:red', marker='*', s=marker_size, zorder=10)
                if best_parameters is not None:
                    ax[i,j].axhline(y=best_parameters[params[i].name], color='tab:red', linestyle='--')
                    ax[i,j].axvline(x=best_parameters[params[j].name], color='tab:red', linestyle='--')
                    # ax[i, j].scatter(best_parameters[params[j].name], best_parameters[params[i].name], c='tab:orange', marker='*', s=marker_size, zorder=10)
                if params[j].axis_type == 'log':
                    ax[i, j].set_xscale('log')
                if params[i].axis_type == 'log':
                    ax[i, j].set_yscale('log')
                ax[i, j].set_xlabel(params[j].display_name + ' [' +params[j].unit+']')
                ax[i, j].set_ylabel(params[i].display_name + ' [' +params[i].unit+']')
            else:
                ax[i, j].set_visible(False)

            #xlim
            ax[i,j].set_xlim(params[j].bounds[0], params[j].bounds[1])
            #ylim
            if i != j:
                ax[i,j].set_ylim(params[i].bounds[0], params[i].bounds[1])
            if j > 0:
                if i != j:
                    ax[i, j].set_yticklabels([])
                    ax[i, j].set_yticklabels([],minor=True)
                    # remove the y axis label
                    ax[i, j].set_ylabel('')

            if i < n - 1:
                ax[i, j].set_xticklabels([])
                ax[i, j].set_xticklabels([],minor=True)
                # remove the x axis label
                ax[i, j].set_xlabel('')

            if i == n - 1:
                ax[i, j].set_xlabel(params[j].display_name + ' [' +params[j].unit+']')
                # rotate x axis label
                ax[i, j].tick_params(axis='x', rotation=45, which='both')
                if j != 0:
                    ax[i, j].set_yticklabels([])
                    ax[i, j].set_yticklabels([],minor=True)
                    ax[i, j].set_ylabel('')

            if j == 0:
                ax[i, j].set_ylabel(params[i].display_name + ' [' +params[i].unit+']')

            if i == j:
                # if i == 0:
                ax[i, j].set_title(params[i].display_name + ' [' +params[i].unit+']')
                    # remove y axis label
                ax[i, j].set_ylabel('P('+params[i].display_name + '|Data)')
                # put y tick labels on the right, only move the label
                ax[i, j].yaxis.set_label_position('right')
                ax[i, j].yaxis.tick_right()
                ax[i, j].yaxis.set_tick_params(which='both', direction='in', left=True, right=True)
                ax[i, j].tick_params(axis='y', labelleft=False, labelright=True)
                # ax[i, j].spines['right'].set_visible(True)
                # ax[i, j].spines['left'].set_visible(True)

    #custim legend 
    # add star for the original parameters
    legend_elements = []
    if params_orig is not None:
        legend_elements.append(plt.Line2D([0], [0], color='k', label='Original parameters', linestyle='-'))
    if best_parameters is not None:
        legend_elements.append(plt.Line2D([0], [0], color='tab:red', label='Best parameters', linestyle='--'))
    if len(legend_elements) > 0:
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.9, 0.5), ncol=1)
    # change spacing between subplots
    plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)

    return fig, ax


def plot_1d_posteriors(params, Nres, objective_name, model, loss, best_parameters= None, params_orig = None, optimizer_type = 'ax',**kwargs):
    """Generate 1D posterior plots for each parameter.

    Parameters
    ----------
    params : list of FitParam() objects
        List of parameters to explore.
    Nres : int
        Number of points to evaluate for each parameter.
    objective_name : str
        Name of the objective to evaluate.
    model : torch model
        Model to evaluate the objective.
    loss : str
        Loss function used, see optimpv/general/general.py for available loss functions.
    best_parameters : dict, optional
        Dictionary of the best parameters, by default None.
    params_orig : dict, optional
        Dictionary of the original parameters, by default None.
    optimizer_type : str, optional
        Type of optimizer used, by default 'ax'.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """    

    n = kwargs.get('n', len(params))
    l = kwargs.get('l', 1)
    ylim = kwargs.get('ylim', None)

    fig_size = kwargs.get('fig_size', (16, 9))
    if n > 3:
        l = int(np.ceil(n/3))

    grid_MSE, dims_GP, dims = get_MSE_grid(params, Nres, objective_name, model, loss, optimizer_type = optimizer_type)
    marginal_posteriors = calculate_1d_posteriors(grid_MSE)

    dims_GP = dims
    fig, axes = plt.subplots(l, 3, figsize=fig_size)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
            continue
        ax.plot(dims_GP[i], marginal_posteriors[i])
        if params_orig is not None:
            ax.axvline(x=params_orig[params[i].name], color='k', linestyle='-')
        if best_parameters is not None:
            ax.axvline(x=best_parameters[params[i].name], color='tab:red', linestyle='--')
        if params[i].axis_type == 'log':
            ax.set_xscale('log')
        ax.set_xlabel(params[i].display_name + ' [' +params[i].unit+']')
        ax.set_ylabel('P('+params[i].display_name + '|Data)')
        if ylim is not None:
            ax.set_ylim(ylim)
    
    legend_elements = []
    if params_orig is not None:
        legend_elements.append(plt.Line2D([0], [0], color='k', label='Original parameters', linestyle='-'))
    if best_parameters is not None:
        legend_elements.append(plt.Line2D([0], [0], color='tab:red', label='Best parameters', linestyle='--'))
    if len(legend_elements) > 0:
        # put in the center
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 1.02), ncol=2)
        
        
    plt.tight_layout()
    return fig, ax

def get_df_from_ax(params, objective_name, optimizer):
    ax_client = optimizer.ax_client

    data = ax_client.experiment.fetch_data().df
    
    dumdic = {}
    # create a dic with the keys of the parameters
    if isinstance(ax_client.experiment.trials[0], BatchTrial):# check if trial is a BatchTrial
        for key in ax_client.experiment.trials[0].arms[0].parameters.keys():
            dumdic[key] = []
        
        # fill the dic with the values of the parameters
        for i in range(len(ax_client.experiment.trials)):

            if ax_client.experiment.trials[i].status == T.COMPLETED:
                for arm in ax_client.experiment.trials[i].arms:
                    if arm.name in data['arm_name'].values: # only add the arm if it is in the data i.e. if it was completed
                        for key in arm.parameters.keys():
                            dumdic[key].append(arm.parameters[key])       
    else:
        for key in ax_client.experiment.trials[0].arm.parameters.keys():
            dumdic[key] = []

        # fill the dic with the values of the parameters
        for i in range(len(ax_client.experiment.trials)):
            if ax_client.experiment.trials[i].status == T.COMPLETED:
                for key in ax_client.experiment.trials[i].arm.parameters.keys():
                    dumdic[key].append(ax_client.experiment.trials[i].arm.parameters[key])

    
    dumdic['iteration'] = list(data[data['metric_name'] == objective_name]['trial_index'])

    dumdic[objective_name] = list(data[data['metric_name'] == objective_name]['mean'])

    df = pd.DataFrame(dumdic)

    # add iteration column with 
    for par in params:
        if par.name in df.columns:
            if par.rescale or par.force_log:
                if par.value_type == 'int':
                    df[par.name] = df[par.name] * par.stepsize
                elif par.value_type == 'float':
                    if par.force_log:
                        df[par.name] = 10 ** df[par.name]
                    else:
                        df[par.name] = df[par.name] * par.fscale
                else: 
                    raise ValueError('Trying to rescale a parameter that is not int or float')
    return df

def plot_density_exploration(params, objective_name, optimizer = None, best_parameters = None, params_orig = None, optimizer_type = 'ax', **kwargs):
    """Generate density plots to visualize the exploration of parameter space.

    Parameters
    ----------
    params : list of FitParam() objects
        List of parameters to explore.
    objective_name : str
        Name of the objective to evaluate.
    optimizer : object, optional
        Optimizer object, by default None.
    best_parameters : dict, optional
        Dictionary of the best parameters, by default None.
    params_orig : dict, optional
        Dictionary of the original parameters, by default None.
    optimizer_type : str, optional
        Type of optimizer used, by default 'ax'.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.

    Raises
    ------
    ValueError
        If the optimizer type is not supported.
    """    

    fig_size = kwargs.get('fig_size', (15, 15))
    levels = kwargs.get('levels', 100)
    
    if optimizer_type == 'ax':
        df = get_df_from_ax(params, objective_name, optimizer)
        
        names = []
        log_scale = []
        axis_limits = []
        for p in params:
            if p.type != 'fixed':
                names.append(p.name)
                log_scale.append(p.axis_type == 'log')
                axis_limits.append(p.bounds)


        # Get all combinations of names
        comb = list(combinations(names, 2))

        # Determine the grid size
        n = len(names)
        fig, axes = plt.subplots(n, n, figsize=fig_size)

        # Plot each combination in the grid
        for i, xx in enumerate(names):
            for j, yy in enumerate(names):
                xval = np.nan
                yval = np.nan
                if params_orig is not None:
                    xval = params_orig[xx]
                    yval = params_orig[yy]

                ax = axes[i, j]
                if i == j:
                    # kde plot on the diagonal
                    try:
                        sns.kdeplot(x=yy, data=df, ax=ax, fill=True, thresh=0, levels=levels, cmap="rocket", color="#03051A", log_scale=log_scale[names.index(xx)])
                    except:
                        # hystogram if kdeplot fails
                        sns.histplot(x=yy, data=df, ax=ax, color="#03051A", log_scale=log_scale[names.index(xx)])

                    if params_orig is not None:
                        ax.axvline(x=yval, color='yellow', linestyle='-')
                    if best_parameters is not None:
                        ax.axvline(x=best_parameters[yy], color='r', linestyle='--')
                    # put point at the best value top of the axis
                

                    if log_scale[names.index(yy)]:
                        ax.set_xscale('log')
                        ax.set_xlim(axis_limits[names.index(yy)])
                    else:
                        ax.set_xlim(axis_limits[names.index(yy)])
                    
                    # put x label on the top
                    # except for the last one
                    if i < n - 1:
                        ax.xaxis.set_label_position('top')
                        ax.xaxis.tick_top()

                elif i > j:
                    kind = 'kde'
                    
                    if kind == 'scatter':
                        sns.scatterplot(x=yy, y=xx, data=df, ax=ax, color="#03051A")
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                    else:
                        try:
                            sns.kdeplot(x=yy, y=xx, data=df, ax=ax, fill=True, thresh=0, levels=levels, cmap="rocket", color="#03051A", log_scale=(log_scale[names.index(yy)], log_scale[names.index(xx)]))
                        except:
                            sns.scatterplot(x=yy, y=xx, data=df, ax=ax, color="#03051A")


                    # Plot as line over the full axis
                    if params_orig is not None:
                        ax.axhline(y=params_orig[xx], color='yellow', linestyle='-')
                        ax.axvline(x=params_orig[yy], color='yellow', linestyle='-')
                    if best_parameters is not None:
                        ax.axhline(y=best_parameters[xx], color='r', linestyle='--')
                        ax.axvline(x=best_parameters[yy], color='r', linestyle='--')
                    
                    ax.set_xlim(axis_limits[names.index(yy)])
                    ax.set_ylim(axis_limits[names.index(xx)])
                else:
                    ax.set_visible(False)

                if j > 0:
                    if i != j:
                        ax.set_yticklabels([])
                        ax.set_yticklabels([],minor=True)
                        # remove the y axis label
                        ax.set_ylabel('')
                if i < n - 1:
                    ax.set_xticklabels([])
                    ax.set_xticklabels([],minor=True)
                    # remove the x axis label
                    ax.set_xlabel('')

                if i == n - 1:
                    for p in params:
                        if p.name == yy:
                            ax.set_xlabel(p.display_name + ' (' + p.unit + ')')
                    ax.tick_params(axis='x', rotation=45, which='both')
                if j == 0:
                    for p in params:
                        if p.name == xx:
                            ax.set_ylabel(p.display_name + ' (' + p.unit + ')')
                if i == j:
                    ax.set_title(params[i].display_name + ' [' +params[i].unit+']')
                    ax.set_ylabel('Density')
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.tick_right()
                    ax.yaxis.set_tick_params(which='both', direction='in', left=True, right=True)
                    ax.tick_params(axis='y', labelleft=False, labelright=True)

        #custom legend 
        legend_elements = []
        if params_orig is not None:
            legend_elements.append(plt.Line2D([0], [0], color='yellow', label='Original parameters', linestyle='-'))
        if best_parameters is not None:
            legend_elements.append(plt.Line2D([0], [0], color='r', label='Best parameters', linestyle='--'))
        if len(legend_elements) > 0:
            fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.9, 0.5), ncol=1)

        # change spacing between subplots
        plt.tight_layout()
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)
    else:
        raise ValueError('This optimizer type is not supported')
    
    return fig, axes

def plot_1D_2D_posterior(params, param_x, param_y, Nres, objective_name, model, loss, best_parameters=None, params_orig=None, optimizer_type='ax', **kwargs):
    """Generate a combined 2D and 1D posterior plot for a specific combination of 2 parameters.

    Parameters
    ----------
    params : list of FitParam() objects
        List of parameters to explore.
    param_x : str
        Name of the parameter to plot on the x-axis.
    param_y : str
        Name of the parameter to plot on the y-axis.
    Nres : int
        Number of points to evaluate for each parameter.
    objective_name : str
        Name of the objective to evaluate.
    model : torch model
        Model to evaluate the objective.
    loss : str
        Loss function used, see optimpv/general/general.py for available loss functions.
    best_parameters : dict, optional
        Dictionary of the best parameters, by default None.
    params_orig : dict, optional
        Dictionary of the original parameters, by default None.
    optimizer_type : str, optional
        Type of optimizer used, by default 'ax'.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    matplotlib.axes._subplots.AxesSubplot
        The axes object containing the plot.
    """    
    fig_size = kwargs.get('fig_size', (12, 12))
    marker_size = kwargs.get('marker_size', 200)
    levels = kwargs.get('levels', Nres)

    grid_MSE, dims_GP, dims  = get_MSE_grid(params, Nres, objective_name, model, loss, optimizer_type=optimizer_type)
    marginal_posteriors = calculate_1d_posteriors(grid_MSE)
    pairwise_posteriors = calculate_2d_posteriors(grid_MSE)

    param_x_idx = [i for i, param in enumerate(params) if param.name == param_x][0]
    param_y_idx = [i for i, param in enumerate(params) if param.name == param_y][0]

    dims_GP = dims
    fig, ax = plt.subplots(2, 2, figsize=fig_size, gridspec_kw={'height_ratios': [1, 4], 'width_ratios': [4, 1], 'hspace': 0.05, 'wspace': 0.05})

    # 2D posterior plot
    ax[1, 0].contourf(dims_GP[param_x_idx], dims_GP[param_y_idx], pairwise_posteriors[param_x_idx * (len(params) - 1) + param_y_idx - 1].reshape(Nres, Nres).T,levels=levels)
    # set x and y limits
    ax[1, 0].set_xlim([params[param_x_idx].bounds[0], params[param_x_idx].bounds[1]])
    ax[1, 0].set_ylim([params[param_y_idx].bounds[0], params[param_y_idx].bounds[1]])
    if params_orig is not None:
        ax[1,0].axhline(y=params_orig[param_y], color='tab:red', linestyle='-')
        ax[1,0].axvline(x=params_orig[param_x], color='tab:red', linestyle='-')
    if best_parameters is not None:
        ax[1,0].axhline(y=best_parameters[param_y], color='tab:orange', linestyle='--')
        ax[1,0].axvline(x=best_parameters[param_x], color='tab:orange', linestyle='--')
        # ax[1, 0].scatter(best_parameters[param_x], best_parameters[param_y], c='tab:orange', marker='*', s=marker_size, zorder=10)
    if params[param_x_idx].axis_type == 'log':
        ax[1, 0].set_xscale('log')
    if params[param_y_idx].axis_type == 'log':
        ax[1, 0].set_yscale('log')
    ax[1, 0].set_xlabel(params[param_x_idx].display_name + ' [' + params[param_x_idx].unit + ']')
    ax[1, 0].set_ylabel(params[param_y_idx].display_name + ' [' + params[param_y_idx].unit + ']')

    # 1D posterior plot for param_x
    ax[0, 0].plot(dims_GP[param_x_idx], marginal_posteriors[param_x_idx])
    # rotate the x-axis labels
    ax[0, 0].tick_params(axis='x', rotation=45, which='both')
    if params_orig is not None:
        ax[0, 0].axvline(x=params_orig[param_x], color='tab:red', linestyle='-')
    if best_parameters is not None:
        ax[0, 0].axvline(x=best_parameters[param_x], color='tab:orange', linestyle='--')
    if params[param_x_idx].axis_type == 'log':
        ax[0, 0].set_xscale('log')
    # set x lim
    ax[0, 0].set_xlim([params[param_x_idx].bounds[0], params[param_x_idx].bounds[1]])
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_xticklabels([],minor=True)
    ax[0, 0].set_ylabel('P('+params[param_x_idx].display_name + '|Data)')

    # 1D posterior plot for param_y
    ax[1, 1].plot(marginal_posteriors[param_y_idx], dims_GP[param_y_idx])
    # set y lim
    ax[1, 1].set_ylim([params[param_y_idx].bounds[0], params[param_y_idx].bounds[1]])
    if params_orig is not None:
        ax[1, 1].axhline(y=params_orig[param_y], color='tab:red', linestyle='-')
    if best_parameters is not None:
        ax[1, 1].axhline(y=best_parameters[param_y], color='tab:orange', linestyle='--')
    if params[param_y_idx].axis_type == 'log':
        ax[1, 1].set_yscale('log')
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_yticklabels([],minor=True)
    ax[1, 1].set_xlabel('P('+params[param_y_idx].display_name + '|Data)')
    # rotate the x-axis labels
    ax[1, 1].tick_params(axis='x', rotation=45, which='both')
    ax[0, 1].axis('off')
    
    # legend customisation
    legend_elements = []
    if params_orig is not None:
        legend_elements.append(plt.Line2D([0], [0], color='tab:red', linestyle='-', label='Original value'))
    if best_parameters is not None:
        legend_elements.append(plt.Line2D([0], [0], color='tab:orange', linestyle='--', label='Best value'))
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.85))

    plt.tight_layout()
    return fig, ax