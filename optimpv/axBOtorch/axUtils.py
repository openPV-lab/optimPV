"""FitParams class"""
# Note: This class is inspired by the https://github.com/i-MEET/boar/ package
######### Package Imports #########################################################################

import numpy as np

######### Function Definitions ####################################################################

def ConvertParamsAx(params):
    """Convert the params to the format required by the Ax/Botorch library

        Parameters
        ----------
        params : list of Fitparam() objects
            list of Fitparam() objects

        Returns
        -------
        list of dict
            list of dictionaries with the following keys:\\
                'name': string: the name of the parameter\\
                'type': string: 'range' or 'fixed'\\
                'bounds': list of float: the lower and upper bounds of the parameter\\
                
        """ 
    
    ax_params = []
    for param in params:
        if param.value_type == 'float':
            if param.type == 'fixed':
                ax_params.append({'name': param.name, 'type': 'fixed', 'value': param.value, 'value_type': 'float'})
            else:
                if param.force_log:
                    ax_params.append({'name': param.name, 'type': 'range', 'bounds': [np.log10(param.bounds[0]), np.log10(param.bounds[1])], 'value_type': 'float', 'log_scale': False})
                else:
                    if param.log_scale:
                        ax_params.append({'name': param.name, 'type': 'range', 'bounds': [param.bounds[0]/param.fscale, param.bounds[1]/param.fscale], 'value_type': 'float', 'log_scale': True})
                    else:
                        ax_params.append({'name': param.name, 'type': 'range', 'bounds': [param.bounds[0]/param.fscale, param.bounds[1]/param.fscale], 'value_type': 'float'})
        elif param.value_type == 'int':
            if param.type == 'fixed':
                ax_params.append({'name': param.name, 'type': 'fixed', 'value': param.value, 'value_type': 'int'})
            else:
                ax_params.append({'name': param.name, 'type': 'range', 'bounds': [param.bounds[0], param.bounds[1]], 'value_type': 'int'})
        elif param.value_type == 'cat' or param.value_type == 'sub' or param.value_type == 'str': 
            if param.type == 'fixed':
                ax_params.append({'name': param.name, 'type': 'fixed', 'value': param.value, 'value_type': 'str'})
            else:
                ax_params.append({'name': param.name, 'type': 'choice', 'values': param.values, 'value_type': 'str'})
        elif param.value_type == 'bool':
            if param.type == 'fixed':
                ax_params.append({'name': param.name, 'type': 'fixed', 'value': param.value, 'value_type': 'bool'})
            else:
                ax_params.append({'name': param.name, 'type': 'choice', 'values': [True, False], 'value_type': 'bool'})
        else:
            raise ValueError('Failed to convert parameter name: {} to Ax format'.format(param.name))
        
    return ax_params