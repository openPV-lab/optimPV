"""Provides general functionality for Agent objects for non ideal diode simulations"""
######### Package Imports #########################################################################

import os, uuid, sys, copy, warnings
import numpy as np
import pandas as pd
from scipy import interpolate, constants

try: 
    import pvlib
    from pvlib.pvsystem import i_from_v
    got_pvlib = True
except:
    got_pvlib = False
    warnings.warn('pvlib not installed, using scipy for diode equation')

from optimpv import *
from optimpv.general.general import calc_metric, loss_function
from optimpv.general.BaseAgent import BaseAgent
from optimpv.Diodefits.DiodeModel import *

## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')

######### Agent Definition #######################################################################
class DiodeAgent(BaseAgent):
    """Agent object for non ideal diode simulations
    with the following formula:
    J = Jph - J0*[exp(-(V-J*R_series)/(n*Vt*)) - 1] - (V - J*R_series)/R_shunt
    see optimpv.Diodefits.DiodeModel.py for more details

    Parameters
    ----------
    params : list of Fitparam() objects
        List of Fitparam() objects.
    X : array-like
        1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values.
    y : array-like
        1-D array containing the current values.
    exp_format : str or list of str, optional
        Format of the experimental data, by default 'light'.
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
        Name of the agent, by default 'JV'.
    use_pvlib : bool, optional
        If True then use the pvlib library to calculate the diode equation, by default False.
    **kwargs : dict
        Additional keyword arguments.
    """    
    def __init__(self, params, X, y, T = 300, exp_format = 'light', metric = 'mse', loss = 'linear', threshold = 100, minimize = True, yerr = None, weight = None, name = 'diode', use_pvlib = False, **kwargs):
        # super().__init__(**kwargs)

        self.params = params
        self.X = X # voltage and Gfrac
        self.y = y
        self.T = T # temperature in K
        self.exp_format = [exp_format]
        self.metric = [metric]
        self.loss = [loss]
        self.threshold = [threshold]
        self.minimize = [minimize]
        self.yerr = [yerr]
        self.weight = [weight]
        self.name = name
        self.use_pvlib = use_pvlib
        self.kwargs = kwargs

        # check that all elements in exp_format are valid
        for form in self.exp_format:
            if form not in ['dark','light']:
                raise ValueError(f'{form} is an invalid exp_format, must be either "dark" or "light"')

        if got_pvlib == False:
            self.use_pvlib = False
    
    def run(self,parameters):
        """Run the diode model and calculate the loss function

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
        arg_names = ['J0','n','R_series','R_shunt']
        if self.exp_format[0] == 'light':
            arg_names.append('Jph')

        for arg in arg_names:
            if arg not in parameters.keys():
                raise ValueError('Parameter: {} not in parameters dictionary'.format(arg))
        
        if 'T' not in parameters.keys():
            T_ = self.T
        else:
            T_ = parameters['T']

        parameters_rescaled = self.params_rescale(parameters, self.params)
        
        if self.use_pvlib and got_pvlib:
            print('Using pvlib to calculate diode equation')
            nVt = parameters_rescaled['n']*kb*T_
            if self.exp_format[0] == 'dark':
                J = -i_from_v(self.X, 0, parameters_rescaled['J0'], parameters_rescaled['R_series'], parameters_rescaled['R_shunt'], nVt)
            elif self.exp_format[0] == 'light':
                J = -i_from_v(self.X, parameters_rescaled['Jph'], parameters_rescaled['J0'], parameters_rescaled['R_series'], parameters_rescaled['R_shunt'], nVt)
        else:
            if self.exp_format[0] == 'dark':
                J = NonIdealDiode_dark(self.X, parameters_rescaled['J0'], parameters_rescaled['n'], parameters_rescaled['R_series'], parameters_rescaled['R_shunt'], T = T_)

            elif self.exp_format[0] == 'light':
                J = NonIdealDiode_light(self.X, parameters_rescaled['J0'], parameters_rescaled['n'], parameters_rescaled['R_series'], parameters_rescaled['R_shunt'], parameters_rescaled['Jph'], T = T_)

        return J
    

    def run_Ax(self,parameters):
        """Run the diode model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        float
            Loss function value.
        """    
        
        if self.exp_format[0] == 'light':   
            self.compare_logs = self.kwargs.get('compare_logs',False)
        else:
            self.compare_logs = self.kwargs.get('compare_logs',True)
        
        yfit = self.run(parameters) # run the diode model

        dum_dict = {}
        
        for i in range(len(self.exp_format)):
            metric_name = self.metric[i]
            if self.compare_logs:
                epsilon = np.finfo(np.float64).eps
                # if 0 in yfit, then add epsilon to avoid log(0)
                yfit[abs(yfit) <= epsilon] = epsilon
                y_ = copy.deepcopy(self.y)
                y_[abs(y_) <= epsilon] = epsilon
                
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(calc_metric(np.log10(abs(y_)),np.log10(abs(yfit)),sample_weight=self.weight[i],metric_name=metric_name),loss=self.loss[i])
            else:
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(calc_metric(self.y,yfit,sample_weight=self.weight[i],metric_name=metric_name),loss=self.loss[i])

        return dum_dict