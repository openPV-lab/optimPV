"""General functions"""
######### Package Imports #########################################################################

from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, root_mean_squared_error, root_mean_squared_log_error, median_absolute_error
import numpy as np
from scipy.spatial import distance

######### Function Definitions ####################################################################
def calc_metric(y,yfit,sample_weight=None,metric_name='mse'):
    """Calculate the metric between the true values and the predicted values

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True values
    yfit : array-like of shape (n_samples,)
        Predicted values
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights, by default None
    metric_name : str, optional
        Name of the metric to calculate, by default 'mse'  
        Possible values are:
            - 'mse': Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'mape': Mean Absolute Percentage Error
            - 'msle': Mean Squared Log Error
            - 'rmsle': Root Mean Squared Log Error
            - 'rmse': Root Mean Squared Error
            - 'medae': Median Absolute Error
            - 'nrmse': Normalized Root Mean Squared Error
            - 'rmsre': Root Mean Squared Relative Error

    Returns
    -------
    float
        The calculated metric

    Raises
    ------
    ValueError
        If the metric is not implemented
    """    

    # check is nan values are present
    if np.isnan(y).any() or np.isnan(yfit).any():
        return np.nan
    
    if metric_name.lower() == 'mse':
        return mean_squared_error(y, yfit, sample_weight=sample_weight)
    elif metric_name.lower() == 'mae':
        return mean_absolute_error(y, yfit, sample_weight=sample_weight)
    elif metric_name.lower() == 'mape':
        return  mean_absolute_percentage_error(y, yfit, sample_weight=sample_weight)
    elif metric_name.lower() == 'msle':
        return  mean_squared_log_error(y, yfit, sample_weight=sample_weight)
    elif metric_name.lower() == 'rmsle':
        return  root_mean_squared_log_error(y, yfit, sample_weight=sample_weight)
    elif metric_name.lower() == 'rmse':
        return  root_mean_squared_error(y, yfit, sample_weight=sample_weight)
    elif metric_name.lower() == 'medae':
        return  median_absolute_error(y, yfit, sample_weight=sample_weight)
    elif metric_name.lower() == 'nrmse':
        maxi = max(np.max(y),np.max(yfit))
        mini = min(np.min(y),np.min(yfit))
        return  root_mean_squared_error(y, yfit,sample_weight=sample_weight)/(maxi-mini)
    elif metric_name.lower() == 'rmsre':
        epsilon = np.finfo(np.float64).eps
        return  np.sqrt(np.mean(((y-yfit)/np.maximum(np.abs(y),epsilon))**2))
    elif metric_name.lower() == 'maxe':
        return  max_error(y, yfit)    
    else:
        raise ValueError('The metric '+metric_name+' is not implemented.')

def loss_function(value,loss='linear'):
    """Calculate the loss function for the given value. Inspired by the scipy loss functions (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).  
    The following loss functions are implemented:

        * 'linear' (default) : ``rho(z) = z``. Gives a standard
            least-squares problem.
        * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
            approximation of l1 (absolute value) loss. Usually a good
            choice for robust least squares.
        * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
            similarly to 'soft_l1'.
        * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
            influence, but may cause difficulties in optimization process.
        * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
            a single residual, has properties similar to 'cauchy'.
        * 'log' : ``rho(z) = log( z)``. Logarithmically scales the
            loss, very similar to 'cauchy' but not as safe.
        * 'log10' : ``rho(z) = log10(z)``. Logarithmically scales the
            loss with base 10 log, very similar to 'cauchy' but not as safe.

    Parameters
    ----------
    value : float
        value to calculate the loss function
    loss : str, optional
        loss function to use, by default

    Returns
    -------
    float
        value of the loss function

    Raises
    ------
    ValueError
        If the loss function is not implemented
    """    

    if loss.lower() == 'linear' :
        return value
    elif loss.lower() == 'log':
        return np.log(abs(value))
    elif loss.lower() == 'log10':
        return np.log10(abs(value))
    elif loss.lower() == 'soft_l1':
        return 2 * ((1 + value)**0.5 - 1)
    elif loss.lower() == 'cauchy':
        return np.log(1 + value)
    elif loss.lower() == 'arctan':
        return np.arctan(value)
    elif loss.lower() == 'huber':
        if abs(value) <= 1:
            return value
        else:
            return 2 * value**0.5 - 1
    else:
        raise ValueError('The loss '+loss+' is not implemented.')   

def inv_loss_function(value,loss='linear'):
    """Calculate the inverse loss function for the given value. Inspired by the scipy loss functions (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
    The following loss functions are implemented:

        * 'linear' (default) : ``rho(z) = z``. Gives a standard
            least-squares problem.
        * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
            approximation of l1 (absolute value) loss. Usually a good
            choice for robust least squares.
        * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
            similarly to 'soft_l1'.
        * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
            influence, but may cause difficulties in optimization process.
        * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
            a single residual, has properties similar to 'cauchy'.
        * 'log' : ``rho(z) = log( z)``. Logarithmically scales the
            loss, very similar to 'cauchy' but not as safe.
        * 'log10' : ``rho(z) = log10(z)``. Logarithmically scales the
            loss with base 10 log, very similar to 'cauchy' but not as safe.

    Parameters
    ----------
    value : float
        value to calculate the inverse loss function
    loss : str, optional
        loss function to use, by default 'linear'

    Returns
    -------
    float
        value of the inverse loss function

    Raises
    ------
    ValueError
        If the loss function is not implemented
    """    
    if loss.lower() == 'linear' :
        return value
    elif loss.lower() == 'log':
        return np.exp(value)
    elif loss.lower() == 'log10':
        return 10**value
    elif loss.lower() == 'soft_l1':
        return ((1 + value / 2)**2 - 1)
    elif loss.lower() == 'cauchy':
        return np.exp(value) - 1
    elif loss.lower() == 'arctan':
        return np.tan(value)
    elif loss.lower() == 'huber':
        if abs(value) <= 1:
            return value
        else:
            return 0.5 * (value + 1)**2
    else:
        raise ValueError('The loss '+loss+' is not implemented.')   

def mean_min_euclidean_distance(X_true, y_true, X_fit, y_fit):
    """Calculate the minimum euclidean distance between the true and the predicted values

    Parameters
    ----------
    X_true : array-like of shape (n_samples,)
        True values of the X coordinate
    y_true : array-like of shape (n_samples,)
        True values of the y coordinate
    X_fit : array-like of shape (n_samples,)
        Predicted values of the X coordinate
    y_fit : array-like of shape (n_samples,)
        Predicted values of the y coordinate

    Returns
    -------
    float
        The average minimum euclidian distance between the true and the predicted values
    """    
    Xy_true = np.hstack((X_true.reshape(-1,1),y_true.reshape(-1,1)))
    Xy_fit = np.hstack((X_fit.reshape(-1,1),y_fit.reshape(-1,1)))
    dists = []
    for i in range(len(Xy_true)):
        dd = []
        for j in range(len(Xy_fit)):
            if i != j:
                dd.append(distance.euclidean(Xy_true[i], Xy_fit[j]))
        dists.append(np.min(dd))
    return np.mean(dists)

def direct_mean_euclidean_distance(X_true, y_true, X_fit, y_fit):
    """Calculate the mean euclidean distance between the true and the predicted values

    Parameters
    ----------
    X_true : array-like of shape (n_samples,)
        True values of the X coordinate
    y_true : array-like of shape (n_samples,)
        True values of the y coordinate
    X_fit : array-like of shape (n_samples,)
        Predicted values of the X coordinate
    y_fit : array-like of shape (n_samples,)
        Predicted values of the y coordinate

    Returns
    -------
    float
        The average euclidian distance between the true and the predicted values
    """    
    Xy_true = np.hstack((X_true.reshape(-1,1),y_true.reshape(-1,1)))
    Xy_fit = np.hstack((X_fit.reshape(-1,1),y_fit.reshape(-1,1)))
    dists = []
    for i in range(len(Xy_true)):
        dists.append(distance.euclidean(Xy_true[i], Xy_fit[i]))

    return np.mean(dists)

