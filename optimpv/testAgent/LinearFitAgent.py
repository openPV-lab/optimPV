"""Provides general functionality for Agent objects for linear fits of the form y = mx + b"""
######### Package Imports #########################################################################

import os, uuid, sys, copy, warnings
import numpy as np

from optimpv import *
from optimpv.general.general import calc_metric, loss_function
from optimpv.general.BaseAgent import BaseAgent

######### Agent Definition #######################################################################
class LinearFitAgent(BaseAgent):
    """Agent object for linear fits of the form y = mx + b
    Useful for testing the optimization algorithms
    
    Parameters
    ----------
    params : list of Fitparam() objects
        List of Fitparam() objects.
    X : array-like
        1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values.
    y : array-like
        1-D array containing the current values.
    exp_format : str or list of str, optional
        Format of the experimental data, by default 'line'.
    metric : str or list of str, optional
        Metric to evaluate the model, see optimpv.general.calc_metric for options, by default 'mse'.
    loss : str or list of str, optional
        Loss function to use, see optimpv.general.loss_function for options, by default 'linear'.
    threshold : int or list of int, optional
        Threshold value for the loss function used when doing multi-objective optimization, by default 100.
    minimize : bool or list of bool, optional
        If True then minimize the loss function, if False then maximize the loss function (note that if running a fit minize should be True), by default True.
    yerr : array-like or list of array-like, optional
        Errors in the current values, by default None.
    weight : array-like or list of array-like, optional
        Weights used for fitting if weight is None and yerr is not None, then weight = 1/yerr**2, by default None.
    name : str, optional
        Name of the agent, by default 'line'.
    **kwargs : dict
        Additional keyword arguments.
    """    
    def __init__(self, params, X, y, exp_format = 'line', metric = 'mse', loss = 'linear', threshold = 100, minimize = True, yerr = None, weight = None, name = 'line', **kwargs):
        # super().__init__(**kwargs)

        self.params = params
        self.X = X 
        self.y = y
        self.exp_format = [exp_format]
        self.metric = [metric]
        self.loss = [loss]
        self.threshold = [threshold]
        self.minimize = [minimize]
        self.yerr = [yerr]
        self.weight = [weight]
        self.name = name
        self.kwargs = kwargs

        # check that all elements in exp_format are valid
        for form in self.exp_format:
            if form not in ['line']:
                raise ValueError(f'{form} is an invalid exp_format, must be either "dark" or "light"')

    
    def run(self,parameters):
        """Run the linear model 

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        float
            Loss function value.
        """    

        # check that all the arguments are in the parameters dictionary
        arg_names = ['m','b']

        for arg in arg_names:
            if arg not in parameters.keys():
                raise ValueError('Parameter: {} not in parameters dictionary'.format(arg))
        

        parameters_rescaled = self.params_rescale(parameters, self.params)
        
        if len(self.X.shape) == 1:
            yfit = parameters_rescaled['m']*self.X + parameters_rescaled['b']
        else:
            yfit = parameters_rescaled['m']*self.X[:,0] + parameters_rescaled['b']

        return yfit
    

    def run_Ax(self,parameters):
        """Run the linear model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        float
            Loss function value.
        """    
        
        yfit = self.run(parameters) # run the diode model

        dum_dict = {}
        
        for i in range(len(self.exp_format)):
            metric_name = self.metric[i]
            dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(calc_metric(self.y,yfit,sample_weight=self.weight[i],metric_name=metric_name),loss=self.loss[i])

        return dum_dict