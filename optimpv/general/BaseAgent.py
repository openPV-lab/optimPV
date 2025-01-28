"""BaseAgent class for Agent objects"""
######### Package Imports #########################################################################

import os,copy,warnings
import numpy as np

######### Agent Definition #######################################################################
class BaseAgent():
    """ Provides general functionality for Agent objects
    
    """    
    def __init__(self) -> None:
        pass

    
    def params_w(self, parameters, params):
        """Populate the Fitparam() objects with the values from the parameters dictionary

        Parameters
        ----------
        parameters : dict
            dictionary of parameter names and values to populate the Fitparam() objects

        params : list of Fitparam() objects
            list of Fitparam() objects to populate

        Raises
        ------
        ValueError
            If the value_type of the parameter is not 'float', 'int', 'str', 'cat', 'sub' or 'bool'

        Returns
        -------
        list of Fitparam() objects
            list of Fitparam() objects populated with the values from the parameters dictionary
        """    

        for param in params:
            if param.name in parameters.keys():
                if param.value_type == 'float':
                    if param.force_log:
                        param.value = 10**float(parameters[param.name])
                    else:
                        param.value = float(parameters[param.name])*param.fscale
                elif param.value_type == 'int':
                    param.value = parameters[param.name]*param.stepsize
                elif param.value_type == 'str':
                    param.value = str(parameters[param.name])
                elif param.value_type == 'cat' or param.value_type == 'sub':
                    param.value = parameters[param.name]
                elif param.value_type == 'bool':
                    param.value = bool(parameters[param.name])
                else:
                    raise ValueError('Failed to convert parameter name: {} to Fitparam() object'.format(param.name))
                
        return params
    
    def params_rescale(self, parameters, params):
        """Rescale the parameters dictionary to match the Fitparam() objects rescaling 

        Parameters
        ----------
        parameters : dict
            dictionary of parameter names and values to rescale the Fitparam() objects

        params : list of Fitparam() objects
            list of Fitparam() objects to rescale

        Raises
        ------
        ValueError
            If the value_type of the parameter is not 'float', 'int', 'str', 'cat', 'sub' or 'bool'

        Returns
        -------
        dict
            dictionary of parameter names and values rescaled
        """    
        dum_dict = {}
        for param in params:
            if param.name in parameters.keys():
                if param.value_type == 'float':
                    if param.force_log:
                        param.value = 10**float(parameters[param.name])
                        dum_dict[param.name] = 10**float(parameters[param.name])
                    else:
                        param.value = float(parameters[param.name])*param.fscale
                        dum_dict[param.name] = float(parameters[param.name])*param.fscale
                elif param.value_type == 'int':
                    param.value = parameters[param.name]*param.stepsize
                    dum_dict[param.name] = parameters[param.name]*param.stepsize
                elif param.value_type == 'str':
                    param.value = str(parameters[param.name])
                    dum_dict[param.name] = str(parameters[param.name])
                elif param.value_type == 'cat' or param.value_type == 'sub':
                    param.value = parameters[param.name]
                    dum_dict[param.name] = parameters[param.name]
                elif param.value_type == 'bool':
                    param.value = bool(parameters[param.name])
                    dum_dict[param.name] = bool(parameters[param.name])
                else:
                    raise ValueError('Failed to convert parameter name: {} to Fitparam() object'.format(param.name))
                
        return dum_dict

    def params_descale(self, parameters, params):
        """Descale the parameters dictionary to match the Fitparam() objects descaling 

        Parameters
        ----------
        parameters : dict
            dictionary of parameter names and values to descale the Fitparam() objects

        params : list of Fitparam() objects
            list of Fitparam() objects to descale

        Raises
        ------
        ValueError
            If the value_type of the parameter is not 'float', 'int', 'str', 'cat', 'sub' or 'bool'

        Returns
        -------
        dict
            dictionary of parameter names and values descaled
        """    
        dum_dict = {}
        for param in params:
            if param.name in parameters.keys():
                if param.value_type == 'float':
                    if param.force_log:
                        dum_dict[param.name] = np.log10(parameters[param.name])
                    else:
                        dum_dict[param.name] = parameters[param.name]/param.fscale
                elif param.value_type == 'int':
                    dum_dict[param.name] = int(parameters[param.name]/param.stepsize)
                elif param.value_type == 'str':
                    dum_dict[param.name] = str(parameters[param.name])
                elif param.value_type == 'cat' or param.value_type == 'sub':
                    dum_dict[param.name] = parameters[param.name]
                elif param.value_type == 'bool':
                    dum_dict[param.name] = bool(parameters[param.name])
                else:
                    raise ValueError('Failed to convert parameter name: {} to Fitparam() object'.format(param.name))
                
        return dum_dict

                       
                       
