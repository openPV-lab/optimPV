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

    def __init__(self, params, y = None, layers = None, thicknesses=None, activeLayer=None,lambda_start=350, lambda_stop=800,lambda_step=1,x_step =1, mat_dir=None,spectrum_file=None,photopic_file=None,exp_format='Jsc', metric=None, loss=None, threshold=10, minimize=False, name='TM'):
        
        self.params = params
        self.y = y
        self.layers = layers
        self.thicknesses = thicknesses
        self.activeLayer = activeLayer
        self.lambda_start = lambda_start
        self.lambda_stop = lambda_stop
        self.lambda_step = lambda_step
        self.x_step = x_step
        self.mat_dir = mat_dir
        self.spectrum_file = spectrum_file
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

        
        # check that layers, thicknesses and activeLayer and spectrum_file are not None
        if self.layers is None:
            raise ValueError('layers must be defined')
        if self.thicknesses is None:
            raise ValueError('thicknesses must be defined')
        if self.activeLayer is None:
            raise ValueError('activeLayer must be defined')
        if self.spectrum_file is None:
            raise ValueError('spectrum_file must be defined')
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


        Jsc, AVT, LUE = TMM(parameters, self.layers, self.thicknesses, self.lambda_start, self.lambda_stop, self.lambda_step, self.x_step, self.activeLayer, self.spectrum_file, self.mat_dir, self.photopic_file)

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

        Jsc, AVT, LUE = self.run(parameters)

        res_dict = {'Jsc':Jsc,'AVT':AVT,'LUE':LUE}

        dum_dict = {}

        for i in range(len(self.exp_format)):
            if self.loss[i] is None:
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = self.target_metric(res_dict[self.exp_format[i]],self.metric[i])
            else:
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(self.target_metric([self.y[i]],yfit=[res_dict[self.exp_format[i]]],metric_name=self.metric[i]),self.loss[i])
   
        return dum_dict



        

        
