"""Provides general functionality for Agent objects for transfer matrix simulations"""
######### Package Imports #########################################################################

import os, uuid, sys, copy, warnings
import numpy as np
import pandas as pd
from scipy import interpolate, constants

from optimpv import *
from optimpv.general.general import calc_metric, loss_function
from optimpv.general.BaseAgent import BaseAgent
from optimpv.TransferMatrix.TransferMatrixModel import *

## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')

######### Agent Definition #######################################################################
class TransferMatrixAgent(BaseAgent):
    """Initialize the TransferMatrixAgent

        Parameters
        ----------
        params : dict
            Dictionary of parameters.
        y : array-like, optional
            Experimental data, by default None.
        layers : list, optional
            List of material names in the stack, by default None. Note that these names will be used to find the refractive index files in the mat_dir. The filenames must be in the form of 'nk_materialname.txt'.
        thicknesses : list, optional
            List of thicknesses of the layers in the stack in meters, by default None.
        lambda_min : float, optional
            Start wavelength in m, by default 350e-9.
        lambda_max : float, optional
            Stop wavelength in m, by default 800e-9.
        lambda_step : float, optional
            Wavelength step in m, by default 1e-9.
        x_step : float, optional
            Step size for the x position in the stack in m, by default 1e-9.
        activeLayer : int, optional
            Index of the active layer in the stack, i.e. the layer where the generation profile will be calculated. Counting starts at 0, by default None.
        spectrum : string, optional
            Name of the file that contains the spectrum, by default None.
        mat_dir : string, optional
            Path to the directory where the refractive index files and the spectrum file are located, by default None.
        photopic_file : string, optional
            Name of the file that contains the photopic response (must be in the same directory as the refractive index files), by default None.
        exp_format : str or list, optional
            Expected format of the output, by default 'Jsc'.
        metric : str or list, optional
            Metric to be used for optimization, by default None.
        loss : str or list, optional
            Loss function to be used for optimization, by default None.
        threshold : int, float or list, optional
            Threshold value for the loss function, by default 10.
        minimize : bool or list, optional
            Whether to minimize the loss function, by default False.
        name : str, optional
            Name of the agent, by default 'TM'.

        Raises
        ------
        ValueError
            If any of the required parameters are not defined or if there is a mismatch in the lengths of metric, loss, threshold, minimize, and exp_format.
    """    
    def __init__(self, params, y = None, layers = None, thicknesses=None, activeLayer=None,lambda_min=350e-9, lambda_max=800e-9,lambda_step=1e-9,x_step =1e-9, mat_dir=None,spectrum=None,photopic_file=None,exp_format='Jsc', metric=None, loss=None, threshold=10, minimize=False, name='TM'):
    
        self.params = params
        self.y = y
        self.layers = layers
        self.thicknesses = thicknesses
        self.activeLayer = activeLayer
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_step = lambda_step
        self.x_step = x_step
        self.mat_dir = mat_dir
        self.spectrum = spectrum
        self.photopic_file = photopic_file
        self.exp_format = exp_format
        self.metric = metric
        self.loss = loss
        self.threshold = threshold
        self.minimize = minimize
        self.name = name

        if isinstance(metric, str):
            self.metric = [metric]
        if isinstance(loss, str):
            self.loss = [loss]
        if isinstance(threshold, (int,float)):
            self.threshold = [threshold]
        if isinstance(minimize, bool):
            self.minimize = [minimize]
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]

        
        # check that all elements in exp_format are valid
        for form in self.exp_format:
            if form not in ['Jsc','AVT','LUE']:
                raise ValueError('{form} is an invalid impedance format. Possible values are: Jsc, AVT, LUE')
            
        if len(self.metric) != len(self.loss) or len(self.metric) != len(self.threshold) or len(self.metric) != len(self.minimize) or len(self.metric) != len(self.exp_format):
            raise ValueError('metric, loss, threshold, minimize and exp_format must have the same length')
        
        for i in range(len(self.metric)):
            if self.metric[i] is None:
                self.metric[i] = ''

        
        # check that layers, thicknesses and activeLayer and spectrum are not None
        if self.layers is None:
            raise ValueError('layers must be defined')
        if self.thicknesses is None:
            raise ValueError('thicknesses must be defined')
        if self.activeLayer is None:
            raise ValueError('activeLayer must be defined')
        if self.spectrum is None:
            raise ValueError('spectrum must be defined')
        if self.photopic_file is None and ('AVT' in self.exp_format or 'LUE' in self.exp_format):
            raise ValueError('photopic_file must be defined to calculate AVT or LUE')
        if self.mat_dir is None:
            raise ValueError('mat_dir must be defined')
        
        # check that layers, thicknesses have the same length
        if len(self.layers) != len(self.thicknesses):
            raise ValueError('layers and thicknesses must have the same length')
        # check that activeLayer is in layers
        if self.activeLayer > len(self.layers):
            raise ValueError('activeLayer must be in layers')
        
    def target_metric(self,y,yfit=None,metric_name=None):
        """Calculates the target metric based on the metric, loss, threshold and minimize values"""
        if metric_name is None or metric_name == '':
            return y
        else:
            return calc_metric(y,yfit,metric_name=metric_name)
    
    def run(self,parameters):

        parameters_rescaled = self.params_rescale(parameters, self.params)

        Jsc, AVT, LUE = TMM(parameters_rescaled, self.layers, self.thicknesses, self.lambda_min, self.lambda_max, self.lambda_step, self.x_step, self.activeLayer, self.spectrum, self.mat_dir, self.photopic_file)

        return Jsc, AVT, LUE
    
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

        # parameters_rescaled = self.params_rescale(parameters, self.params)
        # print(parameters_rescaled)
        Jsc, AVT, LUE = self.run(parameters)

        res_dict = {'Jsc':Jsc,'AVT':AVT,'LUE':LUE}

        dum_dict = {}

        for i in range(len(self.exp_format)):
            if self.loss[i] is None:
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = self.target_metric(res_dict[self.exp_format[i]],self.metric[i])
            else:
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(self.target_metric([self.y[i]],yfit=[res_dict[self.exp_format[i]]],metric_name=self.metric[i]),self.loss[i])
   
        return dum_dict






