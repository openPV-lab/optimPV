import numpy as np
from skopt.space import Real, Integer, Categorical


def ConvertParamsSkOpt(params):
    """Convert the params to the format required by the Scikit-Optimize library ()

        Parameters
        ----------
        params : list of Fitparam() objects
            list of Fitparam() objects

        Returns
        -------
        
                
        """ 
    
    sk_params = []
    for param in params:
        if param.value_type == 'float':
            if param.type != 'fixed':
                if param.force_log:
                    sk_params.append(Real(np.log10(param.bounds[0]), np.log10(param.bounds[1]), name=param.name))
                else:
                    if param.log_scale:
                        sk_params.append(Real(param.bounds[0]/param.fscale, param.bounds[1]/param.fscale, name=param.name, prior='log-uniform'))
                    else:
                        sk_params.append(Real(param.bounds[0]/param.fscale, param.bounds[1]/param.fscale, name=param.name))
        elif param.value_type == 'int':
            if param.type != 'fixed':
                sk_params.append(Integer(int(param.bounds[0]/param.stepsize), int(param.bounds[1]/param.stepsize), name=param.name))
        elif param.value_type == 'cat' or param.value_type == 'sub' or param.value_type == 'str': 
            if param.type != 'fixed':
                sk_params.append(Categorical(param.values, name=param.name))
        elif param.value_type == 'bool':
            raise ValueError('Boolean parameters are not supported by Scikit-Optimize try setting the value_type to cat')
        else:
            raise ValueError('Failed to convert parameter name: {} to SkOpt format'.format(param.name))
        
    return sk_params