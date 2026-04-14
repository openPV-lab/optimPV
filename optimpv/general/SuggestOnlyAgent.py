"""Provides general functionality for Agent objects for non ideal diode simulations"""
######### Package Imports #########################################################################

import os, uuid, sys, copy, warnings
import numpy as np
import pandas as pd
from scipy import interpolate, constants

from optimpv import *
from optimpv.general.general import calc_metric, loss_function, transform_data
from optimpv.general.BaseAgent import BaseAgent




######### Agent Definition #######################################################################
class SuggestOnlyAgent(BaseAgent):
    """Agent object for suggesting new points in an optimization process without running the agents.
 
    Parameters
    ----------
    params : list of Fitparam() objects
        List of Fitparam() objects.
    exp_format : str or list of str, optional
        Format of the experimental data, by default [].
    metric : str or list of str, optional
        Metric to evaluate the model, see optimpv.general.calc_metric for options, by default 'mse'.
    loss : str or list of str, optional
        Loss function to use, see optimpv.general.loss_function for options, by default 'linear'.
    threshold : int or list of int, optional
        Threshold value for the loss function used when doing multi-objective optimization, by default 100.
    minimize : bool or list of bool, optional
        If True then minimize the loss function, if False then maximize the loss function (note that if running a fit minize should be True), by default True.
    tracking_metric : str or list of str, optional
        Additional metrics to track and report in run_Ax output, by default None.
    tracking_loss : str or list of str, optional
        Loss functions to apply to tracking metrics, by default None.
    tracking_exp_format : str or list of str, optional
        Experimental formats for tracking metrics, by default None.
        Weights for tracking metrics, by default None.
    name : str, optional
        Name of the agent, by default 'diode'.
    **kwargs : dict
        Additional keyword arguments.
    """    
    def __init__(self, params, exp_format = None, metric = None, loss = None, 
                threshold = 100, minimize = True,
                tracking_metric = None, tracking_loss = None, tracking_exp_format = None,
                name = 'suggest',  **kwargs):
        # super().__init__(**kwargs)

        self.params = params

       
        # Convert single values to lists for consistency
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]
        else:
            if exp_format is None:
                raise ValueError('exp_format must be provided as a string or a list of strings')
            self.exp_format = exp_format
            
        if isinstance(metric, str):
            self.metric = [metric]
        else:
            if metric is None:
                self.metric = [None]*len(self.exp_format)
            else:
                self.metric = metric
            
        if isinstance(loss, str):
            self.loss = [loss]
        else:
            if loss is None:
                self.loss = [None]*len(self.exp_format)
            else:
                self.loss = loss

        if isinstance(threshold, (int, float)):
            self.threshold = [threshold]
        else:
            self.threshold = threshold
            
        if isinstance(minimize, bool):
            self.minimize = [minimize]
        else:
            self.minimize = minimize
            
        self.name = name
        self.kwargs = kwargs
        self.all_agent_metrics = self.get_all_agent_metric_names()

        # Initialize tracking parameters
        self.tracking_metric = tracking_metric
        self.tracking_loss = tracking_loss
        self.tracking_exp_format = tracking_exp_format

        # Process tracking metrics and losses
        if self.tracking_metric is not None:
            if isinstance(self.tracking_metric, str):
                self.tracking_metric = [self.tracking_metric]
            
            if self.tracking_loss is None:
                self.tracking_loss = [None] * len(self.tracking_metric)
            elif isinstance(self.tracking_loss, str):
                self.tracking_loss = [self.tracking_loss] * len(self.tracking_metric)
                
            # Ensure tracking_metric and tracking_loss have the same length
            if len(self.tracking_metric) != len(self.tracking_loss):
                raise ValueError('tracking_metric and tracking_loss must have the same length')
                
            # Process tracking_exp_format
            if self.tracking_exp_format is None:
                # Default to the main experiment formats if not specified
                self.tracking_exp_format = self.exp_format
            elif isinstance(self.tracking_exp_format, str):
                self.tracking_exp_format = [self.tracking_exp_format]

        self.all_agent_tracking_metrics = self.get_all_agent_tracking_metric_names(add_tracking_suffix=False)




