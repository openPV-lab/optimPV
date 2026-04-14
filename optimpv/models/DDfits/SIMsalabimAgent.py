"""Provides general functionality for Agent objects for SIMsalabim simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy, shutil
from scipy import interpolate
from abc import abstractmethod

from optimpv import *
from optimpv.general.general import calc_metric, loss_function, transform_data
from optimpv.general.BaseAgent import BaseAgent
from pySIMsalabim import *

######### Agent Definition #######################################################################
class SIMsalabimAgent(BaseAgent):
    """ Provides general functionality for Agent objects for SIMsalabim simulations
    
    """
    
    def __init__(self, params, X, y, session_path, simulation_setup=None, exp_format=None, 
                 metric='mse', loss='linear', threshold=100, minimize=True, yerr=None, weight=None, 
                 tracking_metric=None, tracking_loss=None, tracking_exp_format=None, 
                 tracking_X=None, tracking_y=None, tracking_weight=None, transforms='linear', tracking_transforms='linear',
                 name='SIMsalabim', **kwargs) -> None:
        
        self.params = params
        if not isinstance(X, (list, tuple)):
            X = [np.asarray(X)]
        if not isinstance(y, (list, tuple)):
            y = [np.asarray(y)]
        self.X = X
        self.y = y
        self.session_path = session_path
        self.simulation_setup = simulation_setup
        self.exp_format = exp_format
        self.metric = metric
        self.loss = loss
        self.threshold = threshold
        self.minimize = minimize
        self.yerr = yerr
        self.weight = weight
        self.tracking_metric = tracking_metric
        self.tracking_loss = tracking_loss
        self.tracking_exp_format = tracking_exp_format
        self.tracking_X = tracking_X
        self.tracking_y = tracking_y
        self.tracking_weight = tracking_weight
        self.transforms = transforms
        self.tracking_transforms = tracking_transforms
        self.name = name
        self.kwargs = kwargs
        
        # Do some checks and set some default values
        if simulation_setup is None:
            self.simulation_setup = os.path.join(session_path,'simulation_setup.txt')
        else:
            self.simulation_setup = simulation_setup

        if self.loss is None:
            self.loss = 'linear'
        if self.metric is None:
            self.metric = 'mse'
        if self.transforms is None:
            self.transforms = 'linear'

        if isinstance(metric, str):
            self.metric = [metric]
        if isinstance(loss, str):
            self.loss = [loss]
        if isinstance(self.transforms, str):
            self.transforms = [self.transforms]
        if isinstance(threshold, (int,float)):
            self.threshold = [threshold]
        if isinstance(minimize, bool):
            self.minimize = [minimize]
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]

        if weight is not None:
            # check that weight has the same length as y
            if not len(weight) == len(self.y):
                raise ValueError('weight must have the same length as y')
            self.weight = []
            for w in weight:
                if isinstance(w, (list, tuple)):
                    self.weight.append(np.asarray(w))
                else:
                    self.weight.append(w)
        else:
            if yerr is not None:
                # check that yerr has the same length as y
                if not len(yerr) == len(self.y):
                    raise ValueError('yerr must have the same length as y')
                self.weight = []
                for yer in yerr:
                    self.weight.append(1/np.asarray(yer)**2)
            else:
                self.weight = [None]*len(y)

        for exp_format in self.exp_format:
            self.validate_exp_format(exp_format)

        # check that exp_format, metric, loss, threshold and minimize have the same length
        if not len(self.exp_format) == len(self.metric) == len(self.loss) == len(self.threshold) == len(self.minimize) == len(self.X) == len(self.y) == len(self.weight) == len(self.transforms):
            print('Length of exp_format: {}, metric: {}, loss: {}, threshold: {}, minimize: {}, X: {}, y: {}, weight: {}, transforms: {}'.format(len(self.exp_format), len(self.metric), len(self.loss), len(self.threshold), len(self.minimize), len(self.X), len(self.y), len(self.weight), len(self.transforms)))
            raise ValueError('exp_format, metric, loss, threshold, minimize, and transforms must have the same length')
        self.all_agent_metrics = self.get_all_agent_metric_names()    

        # check if simulation_setup file exists
        if not os.path.exists(os.path.join(self.session_path,self.simulation_setup)):
            raise ValueError('simulation_setup file does not exist: {}'.format(os.path.join(self.session_path,self.simulation_setup)))
        if os.name != 'nt':
            try:
                dev_par, layers = load_device_parameters(session_path, simulation_setup, run_mode = False)
            except Exception as e:
                raise ValueError('Error loading device parameters check that all the input files are in the right directory. \n Error: {}'.format(e))
        else:
            warning_timeout = self.kwargs.get('warning_timeout', 10)
            exit_timeout = self.kwargs.get('exit_timeout', 60)
            t_wait = 0
            while True: # need this to be thread safe
                try:
                    dev_par, layers = load_device_parameters(session_path, simulation_setup, run_mode = False)
                    break
                except Exception as e:
                    time.sleep(0.002)
                    t_wait = t_wait + 0.002
                    if t_wait > warning_timeout:
                        print('Warning: SIMsalabim is not responding, please check that all the input files are in the right directory')
                    if t_wait > exit_timeout:
                        raise ValueError('Error loading device parameters check that all the input files are in the right directory. \n Error: {}'.format(e))
                    
        self.dev_par = dev_par
        self.layers = layers
        SIMsalabim_params  = {}

        for layer in layers:
            SIMsalabim_params[layer[1]] = ReadParameterFile(os.path.join(session_path,layer[2]))

        self.SIMsalabim_params = SIMsalabim_params
        pnames = list(SIMsalabim_params[list(SIMsalabim_params.keys())[0]].keys())
        pnames = pnames + list(SIMsalabim_params[list(SIMsalabim_params.keys())[1]].keys())
        self.pnames = pnames   

        # Process tracking metrics and losses
        if self.tracking_metric is not None:
            if isinstance(self.tracking_metric, str):
                self.tracking_metric = [self.tracking_metric]
            
            if self.tracking_loss is None:
                self.tracking_loss = ['linear'] * len(self.tracking_metric)
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
                
            # check that all elements in tracking_exp_format are valid
            for form in self.tracking_exp_format:
                self.validate_exp_format(form)

            if isinstance(self.tracking_transforms, str):
                self.tracking_transforms = [self.tracking_transforms] * len(self.tracking_metric)
                

            # Process tracking_X and tracking_y
            # Check if all tracking formats are in main exp_format
            all_formats_in_main = all(fmt in self.exp_format for fmt in self.tracking_exp_format)
            if self.tracking_X is None or self.tracking_y is None:
                
                if not all_formats_in_main:
                    raise ValueError('tracking_X and tracking_y must be provided when tracking_exp_format contains formats not in exp_format')
                
                # Construct tracking_X and tracking_y from main X and y based on matching formats
                self.tracking_X = []
                self.tracking_y = []
                
                for fmt in self.tracking_exp_format:
                    fmt_indices = [i for i, main_fmt in enumerate(self.exp_format) if main_fmt == fmt]
                    if fmt_indices:
                        # Use the first matching format's data
                        idx = fmt_indices[0]
                        self.tracking_X.append(self.X[idx])
                        self.tracking_y.append(self.y[idx])
            
            # Ensure tracking_X and tracking_y are lists
            if not isinstance(self.tracking_X, list):
                self.tracking_X = [self.tracking_X]
            if not isinstance(self.tracking_y, list):
                self.tracking_y = [self.tracking_y]
                
            # Check that tracking_X and tracking_y have the right lengths
            if len(self.tracking_X) != len(self.tracking_exp_format) or len(self.tracking_y) != len(self.tracking_exp_format):
                raise ValueError('tracking_X and tracking_y must have the same length as tracking_exp_format')
            
            # Process tracking_weight
            if self.tracking_weight is None and all_formats_in_main:
                # Use the main weights if available
                self.tracking_weight = []
                if all_formats_in_main:
                    for fmt in self.tracking_exp_format:
                        fmt_indices = [i for i, main_fmt in enumerate(self.exp_format) if main_fmt == fmt]
                        if fmt_indices:
                            idx = fmt_indices[0]
                            self.tracking_weight.append(self.weight[idx])
                        else:
                            self.tracking_weight.append(None)
                else:
                    self.tracking_weight = [None] * len(self.tracking_exp_format)
            elif not isinstance(self.tracking_weight, list):
                self.tracking_weight = [self.tracking_weight]
                
            # Ensure tracking_weight has the right length
            if len(self.tracking_weight) != len(self.tracking_exp_format):
                raise ValueError('tracking_weight must have the same length as tracking_exp_format')

        if tracking_exp_format is not None:
            # check that tracking_exp_format, tracking_metric and tracking_loss have the same length
            if not len(self.tracking_exp_format) == len(self.tracking_metric) == len(self.tracking_loss) == len(self.tracking_transforms):
                raise ValueError('tracking_exp_format, tracking_metric, tracking_loss and tracking_transforms must have the same length')
        self.all_agent_tracking_metrics = self.get_all_agent_tracking_metric_names()
                   
        
    @abstractmethod
    def validate_exp_format(self, exp_format):
        """
        Validate the exp_format. Must be overridden by child classes.
        """
        pass

    @abstractmethod
    def target_metric(self,y,yfit,metric_name, X=None, Xfit=None,weight=None):
        """
        Calculate the target metric. Must be overridden by child classes.
        """
        pass
    
    def transform_metrics(self, y_true, y_fit, metric, X=None, X_fit=None, weight=None, transforms='linear', do_G_frac_transform=False):
        """Calculate the transformed metrics.
        Transformations can be specified as strings such as 'linear', 'normalize' and more or by a list, see transform_data() function for more details. 

        Parameters
        ----------
        y_true : array-like
            True values.
        y_fit : array-like
            Predicted values.
        metric : str
            Name of the metric to calculate.
        X : array-like, optional
            array, by default None
        X_fit : array-like, optional
            array, by default None
        weight : array-like, optional
            Sample weights, by default None
        transforms : str or list, optional
            Transformation(s) to apply, by default 'linear'
        do_G_frac_transform : bool, optional
            Whether to apply G fraction transformation, by default False

        Returns
        -------
        float
            Calculated metric value.
        """        
        # Apply data transformation based on transforms
        if transforms == 'linear' and not do_G_frac_transform:
            metric_value = self.target_metric(
                y_true,
                y_fit,
                metric,
                X,
                X_fit,
                weight=weight
            )
        else:
            # Transform data for each format
            y_true_transformed, y_pred_transformed = transform_data(
                y_true, 
                y_fit, 
                X=X,
                X_pred=X_fit,
                transforms=transforms,
                do_G_frac_transform=do_G_frac_transform
            )
            
            # Calculate metric with transformed data
            metric_value = calc_metric(
                y_true_transformed, 
                y_pred_transformed, 
                sample_weight=weight, 
                metric_name=metric
            )
        return metric_value
    
    def _run_Ax(self, df, reformat_function):
        """Format the simulation results and calculate the metrics for the Ax optimization loop. 
        It will be used by the run_Ax function defined in each agent class and it will return a dictionary with the metric names and values to be used by Ax.

        Parameters
        ----------
        df : dataframe
            Dataframe containing the simulation results.
        reformat_function : function
            Function to reformat the simulation results into the format required for metric calculation, it takes as input the dataframe, the X values and the exp_format and returns Xfit and yfit.

        Returns
        -------
        dict
            Dictionary containing the calculated metrics.
        """        

        if df is np.nan or len(df) == 0:
            dum_dict = {}
            for i in range(len(self.all_agent_metrics)):
                dum_dict[self.all_agent_metrics[i]] = np.nan

                # Add NaN values for tracking metrics
                if self.tracking_metric is not None:
                    for j in range(len(self.all_agent_tracking_metrics)):
                        dum_dict[self.all_agent_tracking_metrics[j]] = np.nan

            return dum_dict
        
        dum_dict = {}
        #check if Agent has do_Gfrac_transform attribute and set it to False if not
        if not hasattr(self, 'do_G_frac_transform'):
            self.do_G_frac_transform = [False]*len(self.exp_format)
        if not hasattr(self, 'tracking_do_G_frac_transform'):
            if self.tracking_metric is not None:
                self.tracking_do_G_frac_transform = [False]*len(self.tracking_exp_format)
            else:
                self.tracking_do_G_frac_transform = None

        # First loop: calculate main metrics for each exp_format
        for i in range(len(self.exp_format)):
            Xfit, yfit = reformat_function(df, self.X[i], self.exp_format[i])
            
            # Apply data transformation based on transforms
            metric_value = self.transform_metrics(
                self.y[i],
                yfit,
                self.metric[i],
                X=self.X[i],
                X_fit=Xfit,
                weight=self.weight[i],
                transforms=self.transforms[i],
                do_G_frac_transform=self.do_G_frac_transform[i]
            )

            dum_dict[self.all_agent_metrics[i]] = loss_function(metric_value, loss=self.loss[i])
        
        # Second loop: calculate all tracking metrics
        if self.tracking_metric is not None:
            for j in range(len(self.all_agent_tracking_metrics)):
                exp_fmt = self.tracking_exp_format[j]
                metric_name = self.tracking_metric[j]
                loss_type = self.tracking_loss[j]
                
                Xfit, yfit = reformat_function(df, self.tracking_X[j], exp_fmt)
                
                # Apply data transformation based on transforms
                metric_value = self.transform_metrics(
                self.tracking_y[j],
                yfit,
                metric_name,
                X=self.tracking_X[j],
                X_fit=Xfit,
                weight=self.tracking_weight[j],
                transforms=self.tracking_transforms[j],
                do_G_frac_transform=self.tracking_do_G_frac_transform[j]
                )

                dum_dict[self.all_agent_tracking_metrics[j]] = loss_function(metric_value, loss=loss_type)

        return dum_dict

    def get_SIMsalabim_clean_cmd_pars(self, parameters):
        """Get the clean cmd_pars list for the SIMsalabim simulation with properly formatted parameters

        Parameters
        ----------
        parameters : dict or list of Fitparam() objects
            dictionary of parameter names and values or list of Fitparam() objects

        Returns
        -------
        list of dict
            list of dictionaries containing the parameters that have a direct match in the SIMsalabim parameter files with the following form {'par': string, 'val': string}
        Raises
        ------
        ValueError
            Parameter name is not defined in the SIMsalabim parameter files. Please check the parameter names.
        ValueError
            Parameter name is defined in both the parameters and cmd_pars. Please remove one of them.
        UserWarning
            Parameter name is not defined in the SIMsalabim parameter files. Please check the parameter names. The optimization will proceed but the parameter will not be used by SIMsalabim.
        """        

        # if parameters is not a dict:
        if not isinstance(parameters, dict):
            dummy_pars = {}
            for param in parameters:
                if param.value_type == 'float':
                    if param.force_log:
                        dummy_pars[param.name] = np.log10(param.value)
                    else:
                        dummy_pars[param.name] = param.value/param.fscale
                elif param.value_type == 'int':
                    # if param.stepsize is not None:
                    dummy_pars[param.name] = param.value#/param.stepsize
                    
                else:
                    dummy_pars[parameters.name] = parameters.value
            # parameters = dummy_pars
        else:
            dummy_pars = copy.deepcopy(parameters)
                
        VarNames,custom_pars,clean_pars = [],[],[]
        
        # check if cmd_pars is in kwargs
        if 'cmd_pars' in self.kwargs:
            cmd_pars = self.kwargs['cmd_pars']
            for cmd_par in cmd_pars:
                if (cmd_par['par'] not in self.SIMsalabim_params['l1'].keys()) and (cmd_par['par'] not in self.SIMsalabim_params['setup'].keys()):
                    custom_pars.append(cmd_par)
                else:
                    clean_pars.append(cmd_par)
                VarNames.append(cmd_par['par'])
        else:
            cmd_pars = []

        custom_pars, clean_pars, VarNames = self.prepare_cmd_pars(dummy_pars, custom_pars, clean_pars, VarNames)
        
        clean_pars = self.energy_level_offsets(custom_pars, clean_pars)
        
        # removing reset of Gamma_pre parameters and converting them to k_direct because SIMsalabim now does the right thing.
        # clean_pars = self.Gamma_pre_reset(clean_pars,custom_pars)
        clean_pars = self.trap_depth_2_energy_level(custom_pars, clean_pars)

        return clean_pars
    
    def get_SIMsalabim_clean_cmd(self, parameters, sim_type='simss'):
        """Get the command line arguments for the SIMsalabim simulation with properly formatted parameters

        Parameters
        ----------
        parameters : dict or list of Fitparam() objects
            dictionary of parameter names and values or list of Fitparam() objects
        sim_type : str, optional
            type of simulation ('simss' or 'zimt'), by default 'simss'

        Returns
        -------
        str
            command line arguments for the SIMsalabim simulation
        """        

        # get the clean cmd_pars
        clean_pars = self.get_SIMsalabim_clean_cmd_pars(parameters)
        self.check_duplicated_parameters(clean_pars)
        # construct the command line arguments
        return construct_cmd(sim_type, clean_pars)
    

    def package_SIMsalabim_files(self, parameters, sim_type, save_path  = None):
        """ Package the SIMsalabim files for the simulation

        Parameters
        ----------
        parameters : dict or list of Fitparam() objects
            dictionary of parameter names and values or list of Fitparam() objects
        sim_type : str
            type of simulation ('simss' or 'zimt')
        save_path : str, optional
            path to save the simulation files, if None, it will be saved in the session_path/tmp_results, by default None

        Raises
        ------
        ValueError
            if sim_type is not 'simss' or 'zimt'
        """        

        # if parameters is not a dict:
        if not isinstance(parameters, dict):
            dummy_pars = {}
            for param in parameters:
                if param.value_type == 'float':
                    dummy_pars[param.name] = param.value/param.fscale
                elif param.value_type == 'int':
                    dummy_pars[param.name] = param.value#/param.stepsize
                else:
                    dummy_pars[parameters.name] = parameters.value
        else:
            dummy_pars = copy.deepcopy(parameters)
                
        VarNames,custom_pars,clean_pars = [],[],[]
        
        # check if cmd_pars is in kwargs
        if 'cmd_pars' in self.kwargs:
            cmd_pars = self.kwargs['cmd_pars']
            for cmd_par in cmd_pars:
                if (cmd_par['par'] not in self.SIMsalabim_params['l1'].keys()) and (cmd_par['par'] not in self.SIMsalabim_params['setup'].keys()):
                    custom_pars.append(cmd_par)
                else:
                    clean_pars.append(cmd_par)
                VarNames.append(cmd_par['par'])
        else:
            cmd_pars = []

        custom_pars, clean_pars, VarNames = self.prepare_cmd_pars(dummy_pars, custom_pars, clean_pars, VarNames)

        clean_pars = self.energy_level_offsets(custom_pars, clean_pars)

        dum_dic= {}
        for cmd in clean_pars:
            dum_dic[cmd['par']] = cmd['val']


        if save_path is None:
            save_path = os.path.join(self.session_path,'tmp_results')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # copy sim_type file from session_path to save_path
        if sim_type == 'simss':
            if os.name == 'nt':
                shutil.copy(os.path.join(self.session_path,'simss.exe'),os.path.join(save_path,'simss.exe'))
            else:
                shutil.copy(os.path.join(self.session_path,'simss'),os.path.join(save_path,'simss'))
        elif sim_type == 'zimt':
            if os.name == 'nt':
                shutil.copy(os.path.join(self.session_path,'zimt.exe'),os.path.join(save_path,'zimt.exe'))
            else:
                shutil.copy(os.path.join(self.session_path,'zimt'),os.path.join(save_path,'zimt'))
        else:
            raise ValueError('sim_type must be either simss or zimt')

        # get all input files that are needed for the simulation and copy them to save_path with their basename
        InputFiles2copy = []
        dev_par_new = copy.deepcopy(self.dev_par)
        for layer in self.layers:
            if layer[1] == 'setup':
                for section in dev_par_new[layer[2]]:   
                    # print(section[1:])
                    for param in section[1:]:
                        if param[0] == 'par':
                            if param[1] in dum_dic.keys():
                                if self.is_inputFile(param[1],dum_dic[param[1]]):
                                    if os.path.basename(dum_dic[param[1]]) != 'none':
                                        InputFiles2copy.append(dum_dic[param[1]])
                                    param[2] = os.path.basename(dum_dic[param[1]])
                                else:
                                    param[2] = dum_dic[param[1]]
                            else:
                                if self.is_inputFile(param[1],param[2]):
                                    if os.path.basename(param[2]) != 'none':
                                        InputFiles2copy.append(param[2])
                                    param[2] = os.path.basename(param[2])
                                else:
                                    param[2] = param[2]
            else:
                for section in dev_par_new[layer[2]]:
                    for param in section[1:]:
                        if param[0] == 'par':
                            if layer[1]+'.'+param[1] in dum_dic.keys():
                                if self.is_inputFile(param[1],dum_dic[layer[1]+'.'+param[1]]):
                                    if os.path.basename(dum_dic[layer[1]+'.'+param[1]]) != 'none':
                                        InputFiles2copy.append(dum_dic[layer[1]+'.'+param[1]])
                                    param[2] = self.convert_parameter_to_basename(param[1],dum_dic[layer[1]+'.'+param[1]])
                                else:
                                    param[2] = dum_dic[layer[1]+'.'+param[1]]
                            else:
                                if self.is_inputFile(param[1],param[2]):
                                    if os.path.basename(param[2]) != 'none':
                                        InputFiles2copy.append(param[2])
                                    param[2] = self.convert_parameter_to_basename(param[1],param[2])
                                else:
                                    param[2] = param[2]

        #copy the simulation setup file
        shutil.copy(os.path.join(self.session_path,self.simulation_setup),os.path.join(save_path,os.path.basename(self.simulation_setup)))
        for file in InputFiles2copy:
            if os.path.isfile(os.path.join(self.session_path,file)):
                shutil.copy(os.path.join(self.session_path,file),os.path.join(save_path,os.path.basename(file)))

        #update keys with new filenames by taking the basename
        dum_dic = {}
        for key in dev_par_new.keys():
            dum_dic[os.path.basename(key)] = dev_par_new[key]
        dev_par_new = dum_dic

        keys = list(dev_par_new.keys())

        for key in dev_par_new.keys():
            with open(os.path.join(save_path,key),'w') as f:
                f.write(devpar_write_to_txt(dev_par_new[key]))

    def ambi_param_transform(self, param, value, cmd_pars, no_transform=False):
        """Transform the ambipolar parameters to the SIMsalabim format  
        example:  
        mu_ions -> mu_anion and mu_cation  
        N_ions -> N_anion and N_cation  
        mu_np -> mu_n and mu_p  
        C_np_bulk -> C_n_bulk and C_p_bulk  
        C_np_int -> C_n_int and C_p_int  

        Parameters
        ----------
        param : Fitparam() object
            Fitparam() object
        value : float
            value of the parameter
        cmd_pars : list of dict
            list of dictionaries with the following form {'par': string, 'val': string} 

        Returns
        -------
        list of dict
            list of dictionaries with the following form {'par': string, 'val': string} 
        """        
        if '.' in param.name:
            layer, par = param.name.split('.')

            if par == 'N_ions': # this is a special case that defines both N_anion and N_cation as the same value
                if param.value_type == 'float' and not no_transform:
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.N_anion', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.N_cation', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.N_anion', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.N_cation', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.N_anion', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.N_cation', 'val': str(value)})
            elif par == 'C_anion': # this is a special case where anions are actually free to move and cations are fixed btu have the same density, since we can't fix the other ones like in the old SIMsalabim version we actually set N_D instead of N_cation 
                if param.value_type == 'float' and not no_transform:
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.N_anion', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.N_D', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.N_anion', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.N_D', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.N_anion', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.N_D', 'val': str(value)})
            elif par == 'C_cation': # this is a special case where cations are actually free to move and anions are fixed but have the same density, since we can't fix the other ones like in the old SIMsalabim version we actually set N_A instead of N_anion 
                if param.value_type == 'float' and not no_transform:
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.N_cation', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.N_A', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.N_cation', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.N_A', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.N_cation', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.N_A', 'val': str(value)})
            elif par == 'mu_ions':
                if param.value_type == 'float' and not no_transform:
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.mu_anion', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.mu_cation', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.mu_anion', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.mu_cation', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.mu_anion', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.mu_cation', 'val': str(value)})
            elif par == 'mu_np':
                if param.value_type == 'float' and no_transform:
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.mu_n', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.mu_p', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.mu_n', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.mu_p', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.mu_n', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.mu_p', 'val': str(value)})
            elif par == 'C_np_bulk':
                if param.value_type == 'float' and no_transform:
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.C_n_bulk', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.C_p_bulk', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.C_n_bulk', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.C_p_bulk', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.C_n_bulk', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.C_p_bulk', 'val': str(value)})
            elif par == 'C_np_int' and no_transform:
                if param.value_type == 'float':
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.C_n_int', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.C_p_int', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.C_n_int', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.C_p_int', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.C_n_int', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.C_p_int', 'val': str(value)})
        
        return cmd_pars

    def energy_level_offsets(self, custom_pars, clean_pars):
        """Convert the energy level offsets to the SIMsalabim format energy levels
        The formats are:
        offset_lX_lY.E_c is the offset from layer X to layer Y conduction band => E_c_layerY = E_c_layerX - offset
        offset_lX_lY.E_v is the offset from layer X to layer Y valence band => E_v_layerY = E_v_layerX - offset
        Egap_lX.E_c is the bandgap energy of layer X => E_c_layerX = E_v_layerX - Egap
        Egap_lX.E_v is the bandgap energy of layer X => E_v_layerX = E_c_layerX + Egap
        offset_W_L.E_c is the offset from the left electrode work function to the conduction band of layer 1 => W_L = E_c_layer1 - offset
        offset_W_L.E_v is the offset from the left electrode work function to the valence band of layer 1 => W_L = E_v_layer1 - offset
        offset_W_R.E_c is the offset from the right electrode work function to the conduction band of the last layer => W_R = E_c_layerN - offset
        offset_W_R.E_v is the offset from the right electrode work function to the valence band of the last layer => W_R = E_v_layerN - offset

        Parameters
        ----------
        custom_pars : list of dict
            list of dictionaries these contain all the parameters that are not explicitely in the SIMsalabim format and not in the ambipolar format (see ambi_param_transform). The dictionaries are of the form {'par': string, 'val': string}  
        clean_pars : list of dict
            list of dictionaries these contain all the parameters that are explicitely in the SIMsalabim format. The dictionaries are of the form {'par': string, 'val': string}

        Returns
        -------
        list of dict
            list of dictionaries these contain all the parameters converted to the SIMsalabim format. The dictionaries are of the form {'par': string, 'val': string}

        Raises
        ------
        ValueError
            Energy level offset between conduction bands must be defined from right to left
        ValueError
            Energy level offset between valence bands must be defined from left to right
        ValueError
            The offset of the work function of the left electrode with respect to the conduction band must be negative
        ValueError
            The offset of the work function of the left electrode with respect to the valence band must be positive
        ValueError
            The offset of the work function of the right electrode with respect to the conduction band must be negative
        ValueError
            The offset of the work function of the right electrode with respect to the valence band must be positive
        """        
        
        # make a deepcopy of self.SIMsalabim_params to avoid mixing the values of the energy levels when running in parallel
        tmp_SIMsalabim_params = copy.deepcopy(self.SIMsalabim_params)

        # search for energy level values defined in clean_pars and add them to the SIMsalabim_params
        for cmd in clean_pars:
            if ('E_c' in cmd['par']) and (not 'offset' in cmd['par']) and (not 'Egap' in cmd['par']):
                layer, par = cmd['par'].split('.')
                tmp_SIMsalabim_params[layer][par] = cmd['val']
            if ('E_v' in cmd['par']) and (not 'offset' in cmd['par']) and (not 'Egap' in cmd['par']):
                layer, par = cmd['par'].split('.')
                tmp_SIMsalabim_params[layer][par] = cmd['val']
 
        Ec_cmd_nrj, Ev_cmd_nrj, Ec_idx_in_stack, Ec_idx_in_cmd_pars, Ev_idx_in_stack, Ev_idx_in_cmd_pars = [],[],[],[],[],[]
        Egap_cmd_nrj, W_L_offset, W_R_offset = [],[],[]
        for idx, cmd in enumerate(custom_pars):
            if '.' in cmd['par'] and 'offset' in cmd['par'] and not 'W_L' in cmd['par'] and not 'W_R' in cmd['par']:
                layer, par = cmd['par'].split('.')
                offset, layer1, layer2 = layer.split('_')
                if par == 'E_c':
                    Ec_idx_in_stack.append(int(layer1[1:]))
                    Ec_idx_in_cmd_pars.append(idx)
                
                if par == 'E_v':
                    Ev_idx_in_stack.append(int(layer1[1:]))
                    Ev_idx_in_cmd_pars.append(idx)

            if '.' in cmd['par'] and 'Egap' in cmd['par']:
                Egap_cmd_nrj.append(cmd)
            
            if '.' in cmd['par'] and 'offset' in cmd['par'] and 'W_L' in cmd['par']:
                W_L_offset.append(cmd)

            if '.' in cmd['par'] and 'offset' in cmd['par'] and 'W_R' in cmd['par']:
                W_R_offset.append(cmd)

        # reoder the Ec and Ev in cmd_pars to match the order in the stack
        dum_array = np.asarray([Ec_idx_in_stack, Ec_idx_in_cmd_pars])
        dum_array = dum_array[:, dum_array[0].argsort()] # sort the array based on the first row
        Ec_cmd_nrj = [custom_pars[dum_array[1][i]] for i in range(len(dum_array[1]))]

        dum_array = np.asarray([Ev_idx_in_stack, Ev_idx_in_cmd_pars])
        dum_array = dum_array[:, dum_array[0].argsort()] # sort the array based on the first row
        Ev_cmd_nrj = [custom_pars[dum_array[1][i]] for i in range(len(dum_array[1]))] 

        # Set the energy levels of the layers
        Ec_cmd_nrj = Ec_cmd_nrj[::-1] #  invert order of cmd_nrj
        for idx, cmd in enumerate(Ec_cmd_nrj):
            layer, par = cmd['par'].split('.')
            offset, layer1, layer2 = layer.split('_')
            # if int(layer1[1:]) <= int(layer2[1:]):
            #     raise ValueError('The energy level offset between conduction bands must be define from right to left so the offset should be defined as offset_'+layer2+'_offset_'+layer1+' instead of offset_'+layer1+'_offset_'+layer2)
            if par == 'E_c':
                Ec_val = float(tmp_SIMsalabim_params[layer1]['E_c']) - float(cmd['val'])
                clean_pars.append({'par': layer2+'.E_c', 'val': str(Ec_val)})
                tmp_SIMsalabim_params[layer2]['E_c'] = str(Ec_val)
        
        for idx, cmd in enumerate(Ev_cmd_nrj):
            layer, par = cmd['par'].split('.')
            offset, layer1, layer2 = layer.split('_')
            # if int(layer1[1:]) >= int(layer2[1:]):
            #     raise ValueError('The energy level offset between valence bands must be define from left to right so the offset should be defined as offset_'+layer1+'_offset_'+layer2+' instead of offset_'+layer2+'_offset_'+layer1)
            if par == 'E_v':
                Ev_val = float(tmp_SIMsalabim_params[layer1]['E_v']) - float(cmd['val'])
                clean_pars.append({'par': layer2+'.E_v', 'val': str(Ev_val)})
                tmp_SIMsalabim_params[layer2]['E_v'] = str(Ev_val)

        # Set the bandgap energy level of the layers
        for idx, cmd in enumerate(Egap_cmd_nrj):
            layer, par = cmd['par'].split('.')
            Egap, layer_ = layer.split('_')
            if par == 'E_c':
                E_v = float(tmp_SIMsalabim_params[layer_]['E_v'])
                E_c = E_v - float(cmd['val'])
                clean_pars.append({'par': layer_+'.E_c', 'val': str(E_c)})
                tmp_SIMsalabim_params[layer_]['E_c'] = str(E_c)
            if par == 'E_v':
                E_c = float(tmp_SIMsalabim_params[layer_]['E_c'])
                E_v = E_c + float(cmd['val'])
                clean_pars.append({'par': layer_+'.E_v', 'val': str(E_v)})
                tmp_SIMsalabim_params[layer_]['E_v'] = str(E_v)

        # finish with the electrode offsets
        for idx, cmd in enumerate(W_L_offset):
            layer, par = cmd['par'].split('.')
            if par == 'E_c':
                if float(cmd['val']) > 0:
                    raise ValueError('The offset of the work function of the left electrode with respect to the conduction band must be negative')
                W_L = float(tmp_SIMsalabim_params['l1']['E_c']) - float(cmd['val'])
                clean_pars.append({'par': 'W_L', 'val': str(W_L)})
                tmp_SIMsalabim_params['setup']['W_L'] = str(W_L)
            if par == 'E_v':
                if float(cmd['val']) < 0:
                    raise ValueError('The offset of the work function of the left electrode with respect to the valence band must be positive')
                W_L = float(tmp_SIMsalabim_params['l1']['E_v']) - float(cmd['val'])
                clean_pars.append({'par': 'W_L', 'val': str(W_L)})
                tmp_SIMsalabim_params['setup']['W_L'] = str(W_L)
        
        for idx, cmd in enumerate(W_R_offset):
            layer, par = cmd['par'].split('.')
            keys_list = list(tmp_SIMsalabim_params.keys())
            last_layer = keys_list[-1]
            if par == 'E_c':
                if float(cmd['val']) > 0:
                    raise ValueError('The offset of the work function of the right electrode with respect to the conduction band must be negative')
                W_R = float(tmp_SIMsalabim_params[last_layer]['E_c']) - float(cmd['val'])
                clean_pars.append({'par': 'W_R', 'val': str(W_R)})
                tmp_SIMsalabim_params['l1']['W_R'] = str(W_R)
            if par == 'E_v':
                if float(cmd['val']) < 0:
                    raise ValueError('The offset of the work function of the right electrode with respect to the valence band must be positive')
                W_R = float(tmp_SIMsalabim_params[last_layer ]['E_v']) - float(cmd['val'])
                clean_pars.append({'par': 'W_R', 'val': str(W_R)})
                tmp_SIMsalabim_params['setup']['W_R'] = str(W_R)

        return clean_pars

    # def Gamma_pre_reset(self, cmd_pars, custom_pars):
    #     """Prepare the Langevin pre-factor parameters for the SIMsalabim simulation but use it to calculate k_direct instead of passing preLangevin to SIMsalabim.

    #     Parameters
    #     ----------
    #     cmd_pars : list of dict
    #         list of dictionaries with the following form {'par': string, 'val': string}
    #     custom_pars : list of dict
    #         list of dictionaries with the following form {'par': string, 'val': string}

    #     Returns
    #     -------
    #     list of dict
    #         list of dictionaries with the following form {'par': string, 'val': string}
    #     """ 
    #     tmp_SIMsalabim_params = copy.deepcopy(self.SIMsalabim_params)
    #     clean_pars, Gammas = [], []
    #     for cmd in cmd_pars:
    #         if '.' not in cmd['par'] :
    #             clean_pars.append({'par': cmd['par'], 'val': cmd['val']})
    #             continue
    #         else:
    #             if 'mu_n' in cmd['par'] or 'mu_p' in cmd['par']: # make sure we update the mobilities in the tmp_SIMsalabim_params
    #                 if '.' not in cmd['par']:
    #                     continue
    #                 layer, par = cmd['par'].split('.')
    #                 if par == 'mu_n':
    #                     tmp_SIMsalabim_params[layer]['mu_n'] = cmd['val']
    #                 elif par == 'mu_p':
    #                     tmp_SIMsalabim_params[layer]['mu_p'] = cmd['val']
    #                 clean_pars.append({'par': cmd['par'], 'val': cmd['val']})
    #             else:
    #                 clean_pars.append({'par': cmd['par'], 'val': cmd['val']})
    #     for cmd in custom_pars:
    #         if 'Gamma_pre' in cmd['par']:
    #             Gammas.append({'par': cmd['par'], 'val': cmd['val']})
    #     # now we need to add the Gamma_pre parameters to the clean_pars
    #     for gamma in Gammas:
    #         layer, par = gamma['par'].split('.')
    #         mob_n = tmp_SIMsalabim_params[layer]['mu_n']
    #         mob_p = tmp_SIMsalabim_params[layer]['mu_p']
    #         eps_r = tmp_SIMsalabim_params[layer]['eps_r']
    #         eps_0 = 8.8542e-12  # same as in SIMsalabim
    #         q = 1.6022e-19 # same as in SIMsalabim
    #         k_direct = float(gamma['val']) * q * (float(mob_n) + float(mob_p)) / (float(eps_r) * eps_0)
    #         clean_pars.append({'par': layer+'.k_direct', 'val': str(k_direct)})

    #     return clean_pars
    
    def trap_depth_2_energy_level(self, custom_pars, clean_pars):
        """Convert the trap depth parameters to the SIMsalabim format energy levels
        This need to be done after the energy level offsets have been set
        The formats are:
        lX.E_t_bulk_depth.E_c = depth from conduction band => E_t_bulk = E_c + depth
        lX.E_t_bulk_depth.E_v = depth from valence band => E_t_bulk = E_v - depth
        lX.E_t_int_depth.E_c = depth from conduction band => E_t_int = E_c + depth
        lX.E_t_int_depth.E_v = depth from valence band => E_t_int = E_v - depth

        Parameters
        ----------
        custom_pars : list of dict
            list of dictionaries these contain all the parameters that are not explicitely in the SIMsalabim format. The dictionaries are of the form {'par': string, 'val': string}  
        clean_pars : list of dict
            list of dictionaries these contain all the parameters that are explicitely in the SIMsalabim format. The dictionaries are of the form {'par': string, 'val': string}

        Returns
        -------
        list of dict
            list of dictionaries these contain all the parameters converted to the SIMsalabim format. The dictionaries are of the form {'par': string, 'val': string}

        Raises
        ------
        
        """        
        
        # make a deepcopy of self.SIMsalabim_params to avoid mixing the values of the energy levels when running in parallel
        tmp_SIMsalabim_params = copy.deepcopy(self.SIMsalabim_params)

        # search for energy level values defined in clean_pars and add them to the SIMsalabim_params
        for cmd in clean_pars:
            if ('E_c' in cmd['par']) and (not 'offset' in cmd['par']) and (not 'Egap' in cmd['par'])and (not 'depth' in cmd['par']):
                layer, par = cmd['par'].split('.')
                tmp_SIMsalabim_params[layer][par] = cmd['val']
            if ('E_v' in cmd['par']) and (not 'offset' in cmd['par']) and (not 'Egap' in cmd['par'])and (not 'depth' in cmd['par']):
                layer, par = cmd['par'].split('.')
                tmp_SIMsalabim_params[layer][par] = cmd['val']
        
        for idx, cmd in enumerate(custom_pars):
            if 'E_t_bulk_depth' in cmd['par']:
                layer, par, ref = cmd['par'].split('.')
                if ref == 'E_c':
                    E_c = float(tmp_SIMsalabim_params[layer]['E_c'])
                    E_trap = E_c + float(cmd['val'])
                    clean_pars.append({'par': layer+'.E_t_bulk', 'val': str(E_trap)})
                    tmp_SIMsalabim_params[layer]['E_t_bulk'] = str(E_trap)
                if ref == 'E_v':
                    E_v = float(tmp_SIMsalabim_params[layer]['E_v'])
                    E_trap = E_v - float(cmd['val'])
                    clean_pars.append({'par': layer+'.E_t_bulk', 'val': str(E_trap)})
                    tmp_SIMsalabim_params[layer]['E_t_bulk'] = str(E_trap)
            if 'E_t_int_depth' in cmd['par']:
                layer, par, ref = cmd['par'].split('.')
                if ref == 'E_c':
                    E_c = float(tmp_SIMsalabim_params[layer]['E_c'])
                    E_trap = E_c + float(cmd['val'])
                    clean_pars.append({'par': layer+'.E_t_int', 'val': str(E_trap)})
                    tmp_SIMsalabim_params[layer]['E_t_int'] = str(E_trap)
                if ref == 'E_v':
                    E_v = float(tmp_SIMsalabim_params[layer]['E_v'])
                    E_trap = E_v - float(cmd['val'])
                    clean_pars.append({'par': layer+'.E_t_int', 'val': str(E_trap)})
                    tmp_SIMsalabim_params[layer]['E_t_int'] = str(E_trap)
        return clean_pars
    
    def check_duplicated_parameters(self, cmd_pars):
        """Check if there are duplicated parameters in the cmd_pars

        Parameters
        ----------
        cmd_pars : list of dict
            list of dictionaries with the following form {'par': string, 'val': string}

        Raises
        ------
        ValueError
            There are duplicated parameters in the cmd_pars
        """        
        names = []
        for cmd in cmd_pars:
            if cmd['par'] in names:
                raise ValueError('Parameter '+cmd['par']+' is defined more than once in the cmd_pars. Please remove the duplicates.')
            names.append(cmd['par'])

    def prepare_cmd_pars(self, parameters, custom_pars, clean_pars,VarNames):
        """Prepare the cmd_pars for the SIMsalabim simulation

        Parameters
        ----------
        parameters : dict
            dictionary of parameter names and values
        custom_pars : list of dict
            list of dictionaries containing the custom parameters that do not have a direct match in the SIMsalabim parameter files with the following form {'par': string, 'val': string}
        clean_pars : list of dict
            list of dictionaries containing the parameters that have a direct match in the SIMsalabim parameter files with the following form {'par': string, 'val': string}
        VarNames : list of str
            list of parameter names

        Returns
        -------
        list of dict
            list of dictionaries containing the custom parameters that do not have a direct match in the SIMsalabim parameter files with the following form {'par': string, 'val': string}
        list of dict
            list of dictionaries containing the parameters that have a direct match in the SIMsalabim parameter files with the following form {'par': string, 'val': string}
        list of str
            list of parameter names

        Raises
        ------
        ValueError
            If the parameter name is not in the self.params list
        ValueError
            If the parameter name is in both the parameters and cmd_pars
        """        

        for param in self.params:
            if param.name in parameters.keys():
                if param.name not in VarNames:
                    VarNames.append(param.name)
                    if '.' in param.name and 'offset' not in param.name and 'Egap' not in param.name and not 'depth' in param.name:
                        layer, par = param.name.split('.')
                        if par not in ['N_ions', 'mu_ions', 'mu_np', 'C_np_bulk', 'C_np_int','C_anion','C_cation']:
                            if par in self.SIMsalabim_params[layer].keys():
                                if param.value_type == 'float':
                                    if param.force_log:
                                        clean_pars.append({'par': param.name, 'val': str(10**parameters[param.name])})
                                    else:
                                        clean_pars.append({'par': param.name, 'val': str(parameters[param.name]*param.fscale)})
                                elif param.value_type == 'int':
                                    clean_pars.append({'par': param.name, 'val': str(int(parameters[param.name]))})#*param.stepsize))})
                                else:
                                    clean_pars.append({'par': param.name, 'val': str(parameters[param.name])})
                            else:
                                # put in custom_pars
                                if param.value_type == 'float':
                                    if param.force_log:
                                        custom_pars.append({'par': param.name, 'val': str(10**parameters[param.name])})
                                    else:
                                        custom_pars.append({'par': param.name, 'val': str(parameters[param.name]*param.fscale)})
                                elif param.value_type == 'int':
                                    custom_pars.append({'par': param.name, 'val': str(int(parameters[param.name]))})#*param.stepsize))})
                                else:
                                    custom_pars.append({'par': param.name, 'val': str(parameters[param.name])})
                        else:
                            clean_pars = self.ambi_param_transform(param, parameters[param.name], clean_pars)                      
                    else:
                        if param.name in self.SIMsalabim_params['setup'].keys():
                            if param.value_type == 'float':
                                if param.force_log:
                                    clean_pars.append({'par': param.name, 'val': str(10**parameters[param.name])})
                                else:
                                    clean_pars.append({'par': param.name, 'val': str(parameters[param.name]*param.fscale)})
                            elif param.value_type == 'int':
                                clean_pars.append({'par': param.name, 'val': str(int(parameters[param.name]))})#*param.stepsize))})
                            else:
                                clean_pars.append({'par': param.name, 'val': str(parameters[param.name])})
                        else:
                            # put in custom_pars
                            if 'offset' in param.name or 'Egap' in param.name or 'depth' in param.name:
                                if param.value_type == 'float':
                                    if param.force_log:
                                        custom_pars.append({'par': param.name, 'val': str(10**parameters[param.name])})
                                    else:
                                        custom_pars.append({'par': param.name, 'val': str(parameters[param.name]*param.fscale)})
                                elif param.value_type == 'int':
                                    custom_pars.append({'par': param.name, 'val': str(int(parameters[param.name]))})#*param.stepsize))})
                                else:
                                    custom_pars.append({'par': param.name, 'val': str(parameters[param.name])})
                            else:
                                warnings.warn('Parameter '+param.name+' is not defined in the SIMsalabim parameter files. Please check the parameter names. The optimization will proceed but '+param.name+' will not be used by SIMsalabim.', UserWarning)
                            # raise ValueError('Parameter '+param.name+' is not defined in the SIMsalabim parameter files. Please check the parameter names.')
                        
                else:
                    raise ValueError('Parameter '+param.name+' is defined in both the parameters and cmd_pars. Please remove one of them.')
            else:
                # if param is not in parameters we use the param.value
                if param.name not in VarNames:
                    VarNames.append(param.name)
                    if '.' in param.name and 'offset' not in param.name and 'Egap' not in param.name and not 'depth' in param.name:
                        layer, par = param.name.split('.')
                        if par not in ['N_ions', 'mu_ions', 'mu_np', 'C_np_bulk', 'C_np_int','C_anion','C_cation']:
                            if par in self.SIMsalabim_params[layer].keys():
                                clean_pars.append({'par': param.name, 'val': str(param.value)})
                            else:
                                custom_pars.append({'par': param.name, 'val': str(param.value)})
                        else:
                            clean_pars = self.ambi_param_transform(param, param.value, clean_pars, no_transform=True)
                    else:
                        if param.name in self.SIMsalabim_params['setup'].keys():
                            clean_pars.append({'par': param.name, 'val': str(param.value)})
                        else:
                            if 'offset' in param.name or 'Egap' in param.name or 'depth' in param.name:
                                custom_pars.append({'par': param.name, 'val': str(param.value)})
                            else:
                                warnings.warn('Parameter '+param.name+' is not defined in the SIMsalabim parameter files. Please check the parameter names. The optimization will proceed but '+param.name+' will not be used by SIMsalabim.', UserWarning)
                else:
                    raise ValueError('Parameter '+param.name+' is defined in both the parameters and cmd_pars. Please remove one of them.')
                # raise ValueError('There is no parameter named '+param.name+' in the self.params list. Please check the parameter names.')

        return custom_pars, clean_pars, VarNames
    
    def convert_parameter_to_basename(self, name, value):
        """Convert the parameter value to its basename if it is a file

        Parameters
        ----------
        name : str
            parameter name
        value : str or float or int
            parameter value

        Returns
        -------
        str
            parameter value in its basename
        """        

        if name.endswith('File'):
            return os.path.basename(value)
        
        elif name.startswith('l') and name[1:].isdigit():
            return os.path.basename(value)
        
        elif name == 'genProfile':
            if value.lower() != 'calc' and value.lower() != 'none':
                return os.path.basename(value)
        
        elif name.lower() == 'expjv' and value.lower() != 'none':
            return os.path.basename(value)
        
        elif name.startswith('nk'):
            return os.path.basename(value)
        
        elif name.lower() == 'spectrum':
            return os.path.basename(value)
        
        else:
            return value

    def is_inputFile(self, name, value):
        """Check if the parameter is an input file

        Parameters
        ----------
        name : str
            parameter name
        value : str or float or int
            parameter value

        Returns
        -------
        bool
            True if the parameter is an input file, False otherwise
        """        
        if name.endswith('File'):
            return True
        elif name.startswith('l') and name[1:].isdigit():
            return True
        elif name == 'genProfile':
            if value.lower() != 'calc' and value.lower() != 'none':
                return True
        elif name.lower() == 'expjv' and value.lower() != 'none':
            return True
        elif name.startswith('nk'):
            return True
        elif name.lower() == 'spectrum':
            return True
        else:
            return False    