"""General functions"""
######### Package Imports #########################################################################

from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, root_mean_squared_error, root_mean_squared_log_error, median_absolute_error
import numpy as np
from scipy import interpolate
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
    elif metric_name.lower() == 'nllh':
        LLH = -1/2 * np.sum((y - yfit)**2 * sample_weight + np.log(2 * np.pi * 1/sample_weight))
        return -LLH
    elif metric_name.lower() == 'llh':
        # the following assumes that sample_weight is actually the precision (1/sigma^2)
        LLH = -1/2 * np.sum((y - yfit)**2 * sample_weight + np.log(2 * np.pi * 1/sample_weight))
        return LLH
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
        if type(value) == np.ndarray:
            value = np.asarray(value)
            result = np.where(np.abs(value) <= 1, value, 0.5 * (value + 1)**2)
        else:
            if abs(value) <= 1:
                return value
            else:
                return 0.5 * (value + 1)**2
        return result
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

def transform_data_old(y, y_pred, X=None, X_pred=None, transform_type='linear', epsilon=None, do_G_frac_transform=False):
    """Transform data according to specified transformation type
    
    Parameters
    ----------
    y : array-like
        True values to transform
    y_pred : array-like
        Predicted values to transform alongside y
    X : array-like, optional
        X coordinates of true values, by default None
    X_pred : array-like, optional
        X coordinates of predicted/fitted values, by default None
    transform_type : str, optional
        Type of transformation to apply, by default 'linear'
        Possible values are:
        
            - 'linear': No transformation
            - 'log': Log10 transformation of absolute values
            - 'normalized': Division by maximum value
            - 'normalized_log': Normalization followed by log transformation
            - 'sqrt': Square root transformation
    epsilon : float, optional
        Small value to add to avoid log(0), by default the machine epsilon for float64
    do_G_frac_transform : bool, optional
        Whether to apply a specific transformation based on the second column of X, by default False
        
    Returns
    -------
    tuple of array-like
        (y_transformed, y_pred_transformed)
    
    Raises
    ------
    ValueError
        If the transformation type is not implemented
    """
    # Make deep copies to avoid modifying the original data
    y_transformed = np.copy(y)
    y_pred_transformed = np.copy(y_pred)
    # Set epsilon to machine epsilon if not provided
    if epsilon is None:
        epsilon = np.finfo(np.float64).eps

    if transform_type.lower() == 'linear':
        return y_transformed, y_pred_transformed
    elif transform_type.lower() == 'log':
        # Replace zeros with epsilon to avoid log(0)
        y_transformed = np.abs(y_transformed)
        y_transformed[y_transformed <= 0] = epsilon
        
        y_pred_transformed = np.abs(y_pred_transformed)
        y_pred_transformed[y_pred_transformed <= 0] = epsilon
        
        return np.log10(y_transformed), np.log10(y_pred_transformed)
    elif transform_type.lower() == 'sqrt':
        # Ensure values are non-negative for sqrt
        y_transformed[y_transformed < 0] = 0
        y_pred_transformed[y_pred_transformed < 0] = 0
        
        return np.sqrt(y_transformed), np.sqrt(y_pred_transformed)


    # If G_frac transformation is requested, extract unique G_frac values
    if do_G_frac_transform and X is not None and X.shape[1] >= 2:
        Gfracs, index = np.unique(X[:, 1], return_index=True)
        if len(Gfracs) == 1:
            Gfracs = None 
        else:
            Gfracs = Gfracs[np.argsort(index)]

    if not do_G_frac_transform or Gfracs is None:        
        if transform_type.lower() == 'normalized':
            y_transformed = y_transformed/np.max(y_transformed)  # Normalize to [0, 1]
            y_pred_transformed = y_pred_transformed/np.max(y_pred_transformed)
            return y_transformed, y_pred_transformed
            # # Find the maximum value across both arrays for consistent normalization
            # max_val = max(np.max(np.abs(y_transformed)), np.max(np.abs(y_pred_transformed)))
            # if max_val > 0:  # Avoid division by zero
            #     return y_transformed / max_val, y_pred_transformed / max_val
            # return y_transformed, y_pred_transformed
        
        elif transform_type.lower() == 'normalized_log':
            # First normalize using the combined max value
            # max_val = max(np.max(np.abs(y_transformed)), np.max(np.abs(y_pred_transformed)))
            # if max_val > 0:  # Avoid division by zero
            #     y_transformed = y_transformed / max_val
                # y_pred_transformed = y_pred_transformed / max_val
            y_transformed = y_transformed/np.max(y_transformed)  # Normalize to [0, 1]
            y_pred_transformed = y_pred_transformed/np.max(y_pred_transformed)
            # Then log transform
            y_transformed = np.abs(y_transformed)
            y_transformed[y_transformed <= 0] = epsilon
            y_pred_transformed = np.abs(y_pred_transformed)
            y_pred_transformed[y_pred_transformed <= 0] = epsilon
            
            return np.log10(y_transformed), np.log10(y_pred_transformed)
        else:
            raise ValueError(f'The transformation type {transform_type} is not implemented.')
    else:
        if transform_type.lower() == 'log':
            for G in Gfracs:
                mask = X[:, 1] == G
                y_transformed[mask] = np.abs(y_transformed[mask])
                y_transformed[mask][y_transformed[mask] <= 0] = epsilon
                y_pred_transformed[mask] = np.abs(y_pred_transformed[mask])
                y_pred_transformed[mask][y_pred_transformed[mask] <= 0] = epsilon
                y_transformed[mask] = np.log10(y_transformed[mask])
                y_pred_transformed[mask] = np.log10(y_pred_transformed[mask])
            return y_transformed, y_pred_transformed
        elif transform_type.lower() == 'normalized':
            for G in Gfracs:
                mask = X[:, 1] == G
                if np.max(y_transformed[mask]) > 0:
                    y_transformed[mask] = y_transformed[mask] / np.max(y_transformed[mask])
                else:
                    return np.nan * np.ones_like(y_transformed), np.nan * np.ones_like(y_pred_transformed)
                if np.max(y_pred_transformed[mask]) > 0:
                    y_pred_transformed[mask] = y_pred_transformed[mask] / np.max(y_pred_transformed[mask])
                else:
                    return np.nan * np.ones_like(y_transformed), np.nan * np.ones_like(y_pred_transformed)
            return y_transformed, y_pred_transformed
        elif transform_type.lower() == 'normalized_log':

            for G in Gfracs:
                mask = X[:, 1] == G
                y_transformed[mask] = np.abs(y_transformed[mask])
                y_transformed[mask][y_transformed[mask] <= 0] = epsilon
                y_pred_transformed[mask] = np.abs(y_pred_transformed[mask])
                y_pred_transformed[mask][y_pred_transformed[mask] <= 0] = epsilon
                y_transformed[mask] = y_transformed[mask] / np.max(y_transformed[mask])
                y_pred_transformed[mask] = y_pred_transformed[mask] / np.max(y_pred_transformed[mask])
                y_transformed[mask] = np.log10(y_transformed[mask]) 
                y_pred_transformed[mask] = np.log10(y_pred_transformed[mask])
            return y_transformed, y_pred_transformed
        else:
            raise ValueError(f'The transformation type {transform_type} is not implemented.')
        
def transform_data(y, y_pred, X=None, X_pred=None, transforms='linear', epsilon=None, do_G_frac_transform=None):
    """Transform data according to specified transformation type
    
    Parameters
    ----------
    y : array-like
        True values to transform
    y_pred : array-like
        Predicted values to transform alongside y
    X : array-like, optional
        X coordinates of true values, by default None
    X_pred : array-like, optional
        X coordinates of predicted/fitted values, by default None
    transform_type : str or list of str, optional
        Type of transformation to apply, if a list is provided, transformations are applied sequentially, by default 'linear'
        Possible values are:
        
            - 'linear': No transformation
            - 'log': Log10 transformation of absolute values
            - 'normalize': Division by maximum value
            - 'sqrt': Square root transformation
    epsilon : float, optional
        Small value to add to avoid log(0), by default the machine epsilon for float64
    do_G_frac_transform : bool, optional
        Whether to apply a specific transformation based on the second column of X, by default False
        
    Returns
    -------
    tuple of array-like
        (y_transformed, y_pred_transformed)
    
    Raises
    ------
    ValueError
        If the transformation type is not implemented
    """

    if do_G_frac_transform is None:
        do_G_frac_transform = False 
    # Make deep copies
    y_t = np.copy(y)
    ypred_t = np.copy(y_pred)

    if epsilon is None:
        epsilon = np.finfo(np.float64).eps

    # Coerce to list
    if isinstance(transforms, str):
        transform_list = [transforms.lower()]
    else:
        transform_list = [t.lower() for t in transforms]

    # --- Extract G-fracs if needed ---
    Gfracs = None
    # check first that X.shape[1] can work
    if X.ndim >= 2:
        if do_G_frac_transform and X is not None and X.shape[1] >= 2:
            Gfracs, index = np.unique(X[:, 1], return_index=True)
            if len(Gfracs) == 1:
                Gfracs = None
            else:
                Gfracs = Gfracs[np.argsort(index)]

    # --- Helper transforms ---

    def t_linear(a, b, mask=None):
        return a, b

    def t_log(a, b, mask=None):
        if mask is None:
            a = np.abs(a)
            b = np.abs(b)
            a[a <= 0] = epsilon
            b[b <= 0] = epsilon
            return np.log10(a), np.log10(b)
        else:
            sel_a = np.abs(a[mask])
            sel_b = np.abs(b[mask])
            sel_a[sel_a <= 0] = epsilon
            sel_b[sel_b <= 0] = epsilon
            a[mask] = np.log10(sel_a)
            b[mask] = np.log10(sel_b)
            return a, b

    def t_sqrt(a, b, mask=None):
        if mask is None:
            a = np.maximum(a, 0)
            b = np.maximum(b, 0)
            return np.sqrt(a), np.sqrt(b)
        else:
            a[mask] = np.sqrt(np.maximum(a[mask], 0))
            b[mask] = np.sqrt(np.maximum(b[mask], 0))
            return a, b

    def t_normalized(a, b, mask=None):
        if mask is None:
            return a / np.max(a), b / np.max(b)
        else:
            if np.max(a[mask]) > 0:
                a[mask] = a[mask] / np.max(a[mask])
            else:
                return np.nan * np.ones_like(a), np.nan * np.ones_like(b)
            if np.max(b[mask]) > 0:
                b[mask] = b[mask] / np.max(b[mask])
            else:
                return np.nan * np.ones_like(a), np.nan * np.ones_like(b)
            return a, b
    
    def t_abs(a, b, mask=None):
        if mask is None:
            return np.abs(a), np.abs(b)
        else:
            a[mask] = np.abs(a[mask])
            b[mask] = np.abs(b[mask])
            return a, b
    
    def t_abs_normalized(a, b, mask=None):
        #normalize by the maximum absolute value
        if mask is None:
            max_a = np.max(np.abs(a))
            max_b = np.max(np.abs(b))
            if max_a > 0:
                a = a/max_a
            else:
                a = np.nan * np.ones_like(a)
            if max_b > 0:
                b = b/max_b
            else:
                b = np.nan * np.ones_like(b)
            return a, b
        else:
            max_a = np.max(np.abs(a[mask]))
            max_b = np.max(np.abs(b[mask]))
            if max_a > 0:
                a[mask] = a[mask]/max_a
            else:
                a[mask] = np.nan * np.ones_like(a[mask])
            if max_b > 0:
                b[mask] = b[mask]/max_b
            else:
                b[mask] = np.nan * np.ones_like(b[mask])
            
            return a, b

    # Mapping: normalized_log is intentionally removed
    TRANSFORMS = {
        'linear': t_linear,
        'log': t_log,
        'sqrt': t_sqrt,
        'normalize': t_normalized,
        'abs_normalize': t_abs_normalized,
        'abs': t_abs
    }

    # --- Apply transforms sequentially ---
    for tname in transform_list:
        if tname not in TRANSFORMS:
            raise ValueError(f'Transformation {tname} is not implemented.')

        transform_fn = TRANSFORMS[tname]

        if do_G_frac_transform and Gfracs is not None:
            for G in Gfracs:
                mask = (X[:, 1] == G)
                y_t, ypred_t = transform_fn(y_t, ypred_t, mask=mask)
        else:
            y_t, ypred_t = transform_fn(y_t, ypred_t)

    return y_t, ypred_t

# def interpolation_safe(x,y,xnew):
#     """Interpolate y values at new x values, while safely handling out-of-bounds and NaN values.

#     Parameters
#     ----------
#     x : array-like
#         Original x values corresponding to y
#     y : array-like
#         Original y values to interpolate
#     xnew : array-like
#         New x values at which to interpolate y

#     Returns
#     -------
#     array-like
#         Interpolated y values at xnew, with NaNs for out-of-bounds or invalid inputs
#     """
    
#     # now redo interpolation with the new t_con and spv_con values
#     do_interp = True
#     if len(x) == len(xnew):
#         if np.allclose(x, xnew):
#             do_interp = False

#     if not do_interp:
#         return y
    
#     try:
#         tck = interpolate.splrep(x, y, s=0)
#         ynew = interpolate.splev(xnew, tck, der=0, ext=0)
#     except:

#         f = interpolate.interp1d(x, y, fill_value='extrapolate', kind='linear')
#         ynew = f(xnew)
#     return ynew

def interpolation_safe(x, y, xnew, mode="linear", log_base=10):
    """Interpolate y at xnew.

    Invalid source values are dropped before interpolation.
    Invalid query values are returned as NaN.
    For log modes, interpolation is performed in log space but the output
    is returned on the original y scale.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xnew = np.asarray(xnew, dtype=float)

    if len(x) == len(xnew) and np.allclose(x, xnew, equal_nan=True):
        return y

    source_mask = np.isfinite(x) & np.isfinite(y)
    query_mask = np.isfinite(xnew)

    if mode == "linear":
        xin = x[source_mask]
        yin = y[source_mask]
        xnew_in = xnew[query_mask]
        invert_y = False

    elif mode == "loglin":
        source_mask &= x > 0
        query_mask &= xnew > 0

        xin = np.log10(x[source_mask]) if log_base == 10 else np.log(x[source_mask])
        yin = y[source_mask]
        xnew_in = np.log10(xnew[query_mask]) if log_base == 10 else np.log(xnew[query_mask])
        invert_y = False

    elif mode == "loglog":
        source_mask &= (x > 0) & (y > 0)
        query_mask &= xnew > 0

        xin = np.log10(x[source_mask]) if log_base == 10 else np.log(x[source_mask])
        yin = np.log10(y[source_mask]) if log_base == 10 else np.log(y[source_mask])
        xnew_in = np.log10(xnew[query_mask]) if log_base == 10 else np.log(xnew[query_mask])
        invert_y = True

    else:
        xin = x[source_mask]
        yin = y[source_mask]
        xnew_in = xnew[query_mask]
        invert_y = False

    ynew = np.full(xnew.shape, np.nan, dtype=float)

    if xin.size == 0 or yin.size == 0:
        return ynew

    if xin.size == 1:
        yinterp = np.full(xnew_in.shape, yin[0], dtype=float)
    else:
        order = np.argsort(xin)
        xin = xin[order]
        yin = yin[order]

        unique_x, unique_idx = np.unique(xin, return_index=True)
        xin = unique_x
        yin = yin[unique_idx]

        if xin.size == 1:
            yinterp = np.full(xnew_in.shape, yin[0], dtype=float)
        else:
            try:
                tck = interpolate.splrep(xin, yin, s=0)
                yinterp = interpolate.splev(xnew_in, tck, der=0, ext=0)
            except Exception:
                f = interpolate.interp1d(xin, yin, fill_value="extrapolate", kind="linear")
                yinterp = f(xnew_in)

    if invert_y:
        yinterp = 10**yinterp if log_base == 10 else np.exp(yinterp)

    ynew[query_mask] = yinterp
    return ynew


import numpy as np
from scipy import interpolate


def interpolation_safe2(
    x,
    y,
    xnew,
    mode="linear",
    method="auto",
    log_base=10,
    slope_ratio_threshold=50,
):
    """
    Robust interpolation supporting linear, loglin and loglog modes.

    Parameters
    ----------
    x, y : array-like
        Input data.
    xnew : array-like
        Query points.
    mode : {"linear", "loglin", "loglog"}
    method : {"auto", "pchip", "spline", "linear"}
    log_base : {10, np.e}
    slope_ratio_threshold : float
        Threshold used by auto mode to detect sharp transitions.

    Returns
    -------
    ynew : ndarray
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xnew = np.asarray(xnew, dtype=float)

    if len(x) == len(xnew) and np.allclose(x, xnew, equal_nan=True):
        return y.copy()

    source_mask = np.isfinite(x) & np.isfinite(y)
    query_mask = np.isfinite(xnew)

    invert_y = False

    if mode == "linear":

        xin = x[source_mask]
        yin = y[source_mask]
        xnew_in = xnew[query_mask]

    elif mode == "loglin":

        source_mask &= x > 0
        query_mask &= xnew > 0

        logfun = np.log10 if log_base == 10 else np.log

        xin = logfun(x[source_mask])
        yin = y[source_mask]
        xnew_in = logfun(xnew[query_mask])

    elif mode == "loglog":

        source_mask &= (x > 0) & (y > 0)
        query_mask &= xnew > 0

        logfun = np.log10 if log_base == 10 else np.log

        xin = logfun(x[source_mask])
        yin = logfun(y[source_mask])
        xnew_in = logfun(xnew[query_mask])

        invert_y = True

    else:
        raise ValueError(
            "mode must be 'linear', 'loglin', or 'loglog'"
        )

    ynew = np.full_like(xnew, np.nan, dtype=float)

    if len(xin) == 0:
        return ynew

    # Sort and remove duplicate x values
    order = np.argsort(xin)
    xin = xin[order]
    yin = yin[order]

    xin, idx = np.unique(xin, return_index=True)
    yin = yin[idx]

    if len(xin) == 1:
        ynew[query_mask] = yin[0]
        return ynew

    # ---------------------------------------------------------
    # Automatic method selection
    # ---------------------------------------------------------

    chosen_method = method

    if method == "auto":

        chosen_method = "spline"

        if len(xin) < 5:
            chosen_method = "pchip"

        dy = np.diff(yin)
        dx = np.diff(xin)

        valid = np.abs(dx) > np.finfo(float).eps

        if np.any(valid):

            slopes = np.abs(dy[valid] / dx[valid])

            finite = np.isfinite(slopes)

            if np.any(finite):

                slopes = slopes[finite]

                max_slope = np.max(slopes)
                med_slope = np.median(slopes)

                if med_slope <= 0:
                    med_slope = np.mean(slopes)

                if med_slope > 0:

                    slope_ratio = max_slope / med_slope

                    if slope_ratio > slope_ratio_threshold:
                        chosen_method = "pchip"

        # Monotonic data -> PCHIP is almost always safer
        if np.all(np.diff(yin) >= 0) or np.all(np.diff(yin) <= 0):
            chosen_method = "pchip"
    print(f"Chosen interpolation method: {chosen_method}")
    # ---------------------------------------------------------
    # Interpolate
    # ---------------------------------------------------------

    try:

        if chosen_method == "pchip":

            interp_fun = interpolate.PchipInterpolator(
                xin,
                yin,
                extrapolate=True,
            )

            yinterp = interp_fun(xnew_in)

        elif chosen_method == "spline":

            tck = interpolate.splrep(
                xin,
                yin,
                s=0,
            )

            yinterp = interpolate.splev(
                xnew_in,
                tck,
                der=0,
                ext=0,
            )

        elif chosen_method == "linear":

            interp_fun = interpolate.interp1d(
                xin,
                yin,
                kind="linear",
                fill_value="extrapolate",
                assume_sorted=True,
            )

            yinterp = interp_fun(xnew_in)

        else:

            raise ValueError(
                "method must be 'auto', 'pchip', "
                "'spline', or 'linear'"
            )

    except Exception:

        interp_fun = interpolate.interp1d(
            xin,
            yin,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )

        yinterp = interp_fun(xnew_in)

    if invert_y:

        if log_base == 10:
            yinterp = 10.0**yinterp
        else:
            yinterp = np.exp(yinterp)

    ynew[query_mask] = yinterp

    return ynew