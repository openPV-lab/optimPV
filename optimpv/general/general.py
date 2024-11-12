from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, root_mean_squared_error, root_mean_squared_log_error, median_absolute_error
import numpy as np

def calc_metric(y,yfit,sample_weight=None,metric_name='mse'):

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
    else:
        raise ValueError('The metric '+metric_name+' is not implemented.')

def loss_function(val,loss='linear'):

    if loss.lower() == 'linear' :
        return val
    elif loss.lower() == 'log':
        return np.log(abs(val))
    elif loss.lower() == 'log10':
        return np.log10(abs(val))
    elif loss.lower() == 'soft_l1':
        return 2 * ((1 + val)**0.5 - 1)
    elif loss.lower() == 'cauchy':
        return np.log(1 + val)
    elif loss.lower() == 'arctan':
        return np.arctan(val)
    elif loss.lower() == 'huber':
        if abs(val) <= 1:
            return val
        else:
            return 2 * val**0.5 - 1
    else:
        raise ValueError('The loss '+loss+' is not implemented.')   

def inv_loss_function(val,loss='linear'):
    if loss.lower() == 'linear' :
        return val
    elif loss.lower() == 'log':
        return np.exp(val)
    elif loss.lower() == 'log10':
        return 10**val
    elif loss.lower() == 'soft_l1':
        return ((1 + val / 2)**2 - 1)
    elif loss.lower() == 'cauchy':
        return np.exp(val) - 1
    elif loss.lower() == 'arctan':
        return np.tan(val)
    elif loss.lower() == 'huber':
        if abs(val) <= 1:
            return val
        else:
            return 0.5 * (val + 1)**2
    else:
        raise ValueError('The loss '+loss+' is not implemented.')   
        