"""Utility functions for the Ax/Botorch library"""
# Note: This class is inspired by the https://github.com/i-MEET/boar/ package
######### Package Imports #########################################################################

import numpy as np
import ax
from ax import *
from ax.service.ax_client import ObjectiveProperties

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
            list of dictionaries with the following keys:

                'name': string: the name of the parameter
                'type': string: 'range' or 'fixed'
                'bounds': list of float: the lower and upper bounds of the parameter
                
        """ 
    if params is None:
        raise ValueError('The params argument is None')
    
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
                ax_params.append({'name': param.name, 'type': 'range', 'bounds': [int(param.bounds[0]/param.stepsize), int(param.bounds[1]/param.stepsize)], 'value_type': 'int'})
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

def CreateObjectiveFromAgent(agent):
    """Create the objective function from the agent

        Parameters
        ----------
        agent : Agent() object
            the agent object

        Returns
        -------
        function
            the objective function
        """ 

    objectives = {}
    for i in range(len(agent.metric)):
        if hasattr(agent,'exp_format'):
            objectives[agent.name+'_'+agent.exp_format[i]+'_'+agent.metric[i]] = ObjectiveProperties(minimize=agent.minimize[i], threshold=agent.threshold[i])
        else:
            objectives[agent.name+'_'+agent.metric[i]] = ObjectiveProperties(minimize=agent.minimize[i], threshold=agent.threshold[i])


    return objectives

def search_spaceAx(search_space):
    parameters = []
    for param in search_space:
        if param['type'] == 'range':
            if param['value_type'] == 'int':
                parameters.append(RangeParameter(name=param['name'], parameter_type=ParameterType.INT, lower=param['bounds'][0], upper=param['bounds'][1]))
            else:
                parameters.append(RangeParameter(name=param['name'], parameter_type=ParameterType.FLOAT, lower=param['bounds'][0], upper=param['bounds'][1]))
        elif param['type'] == 'fixed':
            if param['value_type'] == 'int':
                parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.INT, value=param.value))
            elif param['value_type'] == 'str':
                parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.STRING, value=param.value))
            elif param['value_type'] == 'bool':
                parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.BOOL, value=param.value))
            else:
                parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.FLOAT, value=param.value))
        elif param['type'] == 'choice':
            parameters.append(ChoiceParameter(name=param.name, values=param.values, is_ordered=param.is_ordered, is_sorted=param.is_sorted))
        else:
            raise ValueError('The parameter type is not recognized')
    return SearchSpace(parameters=parameters)