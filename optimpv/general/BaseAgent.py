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
                if param.type == 'fixed':
                    param.value = parameters[param.name]
                    continue

                if param.value_type == 'float':
                    if param.force_log:
                        param.value = 10**float(parameters[param.name])
                    else:
                        param.value = float(parameters[param.name])*param.fscale
                elif param.value_type == 'int':
                    param.value = parameters[param.name]#*param.stepsize
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
            if param.type == 'fixed':
                dum_dict[param.name] = param.value
            else:
                if param.name in parameters.keys():
                    if param.value_type == 'float':
                        if param.force_log:
                            param.value = 10**float(parameters[param.name])
                            dum_dict[param.name] = 10**float(parameters[param.name])
                        else:
                            dum_dict[param.name] = float(parameters[param.name])*param.fscale
                    elif param.value_type == 'int':
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
                else:
                    dum_dict[param.name] = param.value
                
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
            if param.type == 'fixed':
                dum_dict[param.name] = param.value
                continue
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
            else:
                if param.value_type == 'float':
                    if param.force_log:
                        dum_dict[param.name] = np.log10(param.value)
                    else:
                        dum_dict[param.name] = param.value/param.fscale
                elif param.value_type == 'int':
                    dum_dict[param.name] = int(param.value/param.stepsize)
                elif param.value_type == 'str':
                    dum_dict[param.name] = str(param.value)
                elif param.value_type == 'cat' or param.value_type == 'sub':
                    dum_dict[param.name] = param.value
                elif param.value_type == 'bool':
                    dum_dict[param.name] = bool(param.value)
                else:
                    raise ValueError('Failed to convert parameter name: {} to Fitparam() object'.format(param.name))
                
                
        return dum_dict
    
    def rescale_dataframe(self, df, params):
        """Rescale the dataframe to match the Fitparam() objects rescaling 

        Parameters
        ----------
        df : DataFrame
            dataframe to rescale

        params : list of Fitparam() objects
            list of Fitparam() objects to rescale

        Returns
        -------
        DataFrame
            dataframe rescaled
        """    
        for param in params:
            if param.type == 'fixed':
                df[param.name] = param.value
                continue
            if param.value_type == 'float':
                if param.force_log:
                    df[param.name] = 10**df[param.name]
                else:
                    df[param.name] = df[param.name]*param.fscale
            elif param.value_type == 'int':
                df[param.name] = df[param.name]#*param.stepsize
            elif param.value_type == 'str':
                df[param.name] = df[param.name].astype(str)
            elif param.value_type == 'cat' or param.value_type == 'sub':
                df[param.name] = df[param.name].astype(str)
            elif param.value_type == 'bool':
                df[param.name] = df[param.name].astype(bool)
            else:
                raise ValueError('Failed to convert parameter name: {} to Fitparam() object'.format(param.name))
                
        return df
    
    def descale_dataframe(self, df, params):
        """Descale the dataframe to match the Fitparam() objects descaling 

        Parameters
        ----------
        df : DataFrame
            dataframe to descale

        params : list of Fitparam() objects
            list of Fitparam() objects to descale

        Returns
        -------
        DataFrame
            dataframe descaled
        """    
        for param in params:
            if param.type == 'fixed':
                continue
            if param.value_type == 'float':
                if param.force_log:
                    df[param.name] = np.log10(df[param.name])
                else:
                    df[param.name] = df[param.name]/param.fscale
            elif param.value_type == 'int':
                df[param.name] = df[param.name]#/param.stepsize
                # convert to integer type
                df[param.name] = df[param.name].astype(int)
            elif param.value_type == 'str':
                df[param.name] = df[param.name].astype(str)
            elif param.value_type == 'cat' or param.value_type == 'sub':
                df[param.name] = df[param.name].astype(str)
            elif param.value_type == 'bool':
                df[param.name] = df[param.name].astype(bool)
            else:
                raise ValueError('Failed to convert parameter name: {} to Fitparam() object'.format(param.name))
                
        return df
    
    def get_all_agent_metric_names(self):
        """Get all metric names from the agent

        Returns
        -------
        list
            List of all metric names from the agent, formatted as 'name_exp_format_metric_loss'

        Raises
        ------
        ValueError
            If no metric or exp_format is defined in the agent
        """        

        all_agent_metrics = []
        
        if hasattr(self, 'exp_format') and self.exp_format is not None:
            num_metrics = len(self.exp_format)
        elif hasattr(self, 'metric') and self.metric is not None:
            num_metrics = len(self.metric)
        else:
            raise ValueError('No metric or exp_format defined in the agent')
        
        for i in range(num_metrics):
            name = ''
            if hasattr(self, 'exp_format'):
                if isinstance(self.exp_format, list):
                    if self.exp_format[i] is not None:
                        name += '_' + self.exp_format[i]
                else:
                    if self.exp_format is not None:
                        name += '_' + self.exp_format
            if hasattr(self, 'metric'):
                if isinstance(self.metric, list):
                    if self.metric[i] is not None:
                        name += '_' + self.metric[i]
                elif self.metric is not None:
                    name += '_' + self.metric
            if hasattr(self, 'loss'):
                if isinstance(self.loss, list):
                    if self.loss[i] is not None:
                        name += '_' + self.loss[i]
                elif self.loss is not None:
                    name += '_' + self.loss

            if self.name is not None:
                name = self.name + name
            else:
                #remove the first underscore if no name is provided
                name = name.lstrip('_')
            all_agent_metrics.append(name)


        return all_agent_metrics
    
    def get_all_agent_tracking_metric_names(self,add_tracking_suffix=True):
        """Get all tracking metric names from the agent

        Parameters
        ----------
        add_tracking_suffix : bool, optional
            Whether to add a '_tracking' suffix to the metric names, by default True

        Returns
        -------
        list
            List of all tracking metric names from the agent, formatted as 'name_exp_format_metric_loss'

        Raises
        ------
        ValueError
            If no metric or exp_format is defined in the agent
        """ 

        all_agent_tracking_metrics = []
        
        if hasattr(self, 'tracking_exp_format') and self.tracking_exp_format is not None:
            num_metrics = len(self.tracking_exp_format)
        elif hasattr(self, 'tracking_metric') and self.tracking_metric is not None:
            num_metrics = len(self.tracking_metric)
        else:
            return None

        for i in range(num_metrics):
            name = ''
            if hasattr(self, 'tracking_exp_format'):
                if isinstance(self.tracking_exp_format, list):
                    if self.tracking_exp_format[i] is not None:
                        name += '_' + self.tracking_exp_format[i]
                else:
                    if self.tracking_exp_format is not None:
                        name += '_' + self.tracking_exp_format
            if hasattr(self, 'tracking_metric'):
                if isinstance(self.tracking_metric, list):
                    if self.tracking_metric[i] is not None:
                        name += '_' + self.tracking_metric[i]
                elif self.tracking_metric is not None:
                    name += '_' + self.tracking_metric
            if hasattr(self, 'tracking_loss'):
                if isinstance(self.tracking_loss, list):
                    if self.tracking_loss[i] is not None:
                        name += '_' + self.tracking_loss[i]
                elif self.tracking_loss is not None:
                    name += '_' + self.tracking_loss
            if self.name is not None:
                name = self.name + name
            else:
                #remove the first underscore if no name is provided
                name = name.lstrip('_')
            if add_tracking_suffix:
                all_agent_tracking_metrics.append(name+'_tracking')
            else:
                all_agent_tracking_metrics.append(name)

        return all_agent_tracking_metrics
    
    def create_metrics_list(self):
        """
        Create a list of all metrics from all agents.
        
        Returns
        -------
        list
            List of metric names
        list
            List of minimize values
        """
        metrics,minimizes = [],[]
        for agent in self.agents:
            for i in range(len(agent.all_agent_metrics)):
                metrics.append(agent.all_agent_metrics[i])
                minimizes.append(agent.minimize[i])
        return metrics,minimizes

    def get_bounds_list(self):
        """Get the bounds list from the agent's parameters

        Returns
        -------
        list
            List of bounds from the agent's parameters
        """         
        bounds = [[],[]]
        for param in self.params:
            if param.type != 'fixed':
                bounds[0].append(param.bounds[0])
                bounds[1].append(param.bounds[1])
        return bounds
    
    def bounds_descale(self, bounds, params):
        """Descale the bounds to match the Fitparam() objects descaling 

        Parameters
        ----------
        bounds : list
            list of bounds to descale

        params : list of Fitparam() objects
            list of Fitparam() objects to descale

        Returns
        -------
        list
            list of bounds descaled
        """    
        descale_bounds = [[],[]]
        j = 0
        for i,param in enumerate(params):
            if param.type != 'fixed':
                if param.value_type == 'float':
                    if param.force_log:
                        descale_bounds[0].append(np.log10(bounds[0][j]))
                        descale_bounds[1].append(np.log10(bounds[1][j]))
                    else:
                        descale_bounds[0].append(bounds[0][j]/param.fscale)
                        descale_bounds[1].append(bounds[1][j]/param.fscale)
                elif param.value_type == 'int':
                    descale_bounds[0].append(bounds[0][j])#/param.stepsize)
                    descale_bounds[1].append(bounds[1][j])#/param.stepsize)
                else:
                    descale_bounds[0].append(bounds[0][j])
                    descale_bounds[1].append(bounds[1][j])
                j += 1
        return descale_bounds
    
    def rescale_array(self, array, params, param_name):
        """Rescale the array to match the Fitparam() objects rescaling 

        Parameters
        ----------
        array : list
            list of values to rescale

        params : list of Fitparam() objects
            list of Fitparam() objects to rescale

        Returns
        -------
        list
            list of values rescaled
        """    
        rescale_array = []
        j = 0
        for i,param in enumerate(params):
            if param.name == param_name:
                if param.type != 'fixed':
                    if param.value_type == 'float':
                        if param.force_log:
                            for val in array:
                                rescale_array.append(10**val)
                        else:
                            for val in array:
                                rescale_array.append(val*param.fscale)
                    elif param.value_type == 'int':
                        for val in array:
                            rescale_array.append(val)#*param.stepsize)
                    else:
                        for val in array:
                            rescale_array.append(val)
                else:
                    for val in array:
                        rescale_array.append(val)
                break
            
        return np.asarray(rescale_array)