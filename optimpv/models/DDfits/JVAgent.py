"""JVAgent class for steady-state JV simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy, re
from scipy import interpolate

from optimpv import *
from optimpv.general.general import calc_metric, loss_function, transform_data
from optimpv.models.DDfits.SIMsalabimAgent import SIMsalabimAgent
from pySIMsalabim import *
from pySIMsalabim.experiments.JV_steady_state import *

######### Agent Definition #######################################################################
class JVAgent(SIMsalabimAgent):  
    """JVAgent class for steady-state JV simulations with SIMsalabim

    Parameters
    ----------
    params : list of Fitparam() objects
        List of Fitparam() objects.
    X : array-like
        1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values.
    y : array-like
        1-D array containing the current values.
    session_path : str
        Path to the session directory.
    simulation_setup : str, optional
        Path to the simulation setup file, if None then use the default file 'simulation_setup.txt'in the session_path directory, by default None.
    exp_format : str or list of str, optional
        Format of the experimental data, by default 'JV'.
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
    tracking_metric : str or list of str, optional
        Additional metrics to track and report in run_Ax output, by default None.
    tracking_loss : str or list of str, optional
        Loss functions to apply to tracking metrics, by default None.
    tracking_exp_format : str or list of str, optional
        Experimental formats for tracking metrics, by default None.
    tracking_X : array-like or list of array-like, optional
        X values for tracking metrics, by default None.
    tracking_y : array-like or list of array-like, optional
        y values for tracking metrics, by default None.
    tracking_weight : array-like or list of array-like, optional
        Weights for tracking metrics, by default None.
    transforms : str or list of str, optional
        Type of transformation to apply to data before metric calculation, if a list is provided, transformations are applied sequentially, see optimpv.general.transform_data for options, by default 'linear'.
    name : str, optional
        Name of the agent, by default 'JV'.
    **kwargs : dict
        Additional keyword arguments.
    """   
    def __init__(self, params, X, y, session_path, simulation_setup = None, exp_format = ['JV'], 
                 metric = ['mse'], loss = ['linear'], threshold = [100], minimize = [True], 
                 yerr = None, weight = None, tracking_metric = None, tracking_loss = None, 
                 tracking_exp_format = None, tracking_X = None, tracking_y = None, tracking_weight = None, transforms = 'linear',
                 name = 'JV', **kwargs):       

        self.params = params
        self.session_path = session_path  
        if simulation_setup is None:
            self.simulation_setup = os.path.join(session_path,'simulation_setup.txt')
        else:
            self.simulation_setup = simulation_setup

        if not isinstance(X, (list, tuple)):
            X = [np.asarray(X)]
        if not isinstance(y, (list, tuple)):
            y = [np.asarray(y)]

        self.X = X
        self.y = y
        self.yerr = yerr
        self.metric = metric
        self.loss = loss
        self.threshold = threshold
        self.minimize = minimize
        self.tracking_metric = tracking_metric
        self.tracking_loss = tracking_loss
        self.tracking_exp_format = tracking_exp_format
        self.tracking_X = tracking_X
        self.tracking_y = tracking_y
        self.tracking_weight = tracking_weight
        self.transforms = transforms

        if self.loss is None:
            self.loss = 'linear'
        if self.metric is None:
            self.metric = 'mse'
        if isinstance(metric, str):
            self.metric = [metric]
        if isinstance(loss, str):
            self.loss = [loss]
        if isinstance(threshold, (int,float)):
            self.threshold = [threshold]
        if isinstance(minimize, bool):
            self.minimize = [minimize]

        self.kwargs = kwargs
        self.name = name
        self.exp_format = exp_format
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]

        # check that all elements in exp_format are valid
        for JV_form in self.exp_format:
            is_valid = (JV_form == 'JV' or re.match(r'^QFLSL-?\d+$', JV_form) or re.match(r'^K_QFLSL-?\d+$', JV_form))
    
            if not is_valid:
                raise ValueError(f'{JV_form} is an invalid JV format. Possible values are: JV or QFLSL followed by an integer (e.g., QFLSL1, QFLSL2, etc.)')
            # if JV_form not in ['JV'] or 'QFLS' not in JV_form:
            #     raise ValueError('{JV_form} is an invalid JV format. Possible values are: JV or QFLS')
        if weight is not None:
            # check that weight has the same length as y
            if not len(weight) == len(y):
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
                if not len(yerr) == len(y):
                    raise ValueError('yerr must have the same length as y')
                self.weight = []
                for yer in yerr:
                    self.weight.append(1/np.asarray(yer)**2)
            else:
                self.weight = [None]*len(y)

        # check that exp_format, metric, loss, threshold and minimize have the same length
        if not len(self.exp_format) == len(self.metric) == len(self.loss) == len(self.threshold) == len(self.minimize) == len(self.X) == len(self.y) == len(self.weight):
            raise ValueError('exp_format, metric, loss, threshold and minimize must have the same length')
        self.all_agent_metrics = self.get_all_agent_metric_names()       

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
                is_valid = (form == 'JV' or re.match(r'^QFLSL-?\d+$', form))
                if not is_valid:
                    raise ValueError(f'{form} is an invalid tracking_exp_format, must be "JV" or "QFLSL" followed by an integer')

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
            if not len(self.tracking_exp_format) == len(self.tracking_metric) == len(self.tracking_loss):
                raise ValueError('tracking_exp_format, tracking_metric and tracking_loss must have the same length')
        self.all_agent_tracking_metrics = self.get_all_agent_tracking_metric_names()
        
        # Are we doing Gfrac transformations?
        self.do_G_frac_transform = self.kwargs.get('do_G_frac_transform', False)
        if 'do_G_frac_transform' in self.kwargs.keys():
            self.kwargs.pop('do_G_frac_transform')
   
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


    def target_metric(self,y,yfit,metric_name, X=None, Xfit=None,weight=None):
        """Calculate the target metric depending on self.metric

        Parameters
        ----------
        y : array-like
            1-D array containing the current values.
        yfit : array-like
            1-D array containing the fitted current values.
        metric_name : str
            Metric to evaluate the model, see optimpv.general.calc_metric for options.
        X : array-like, optional
            1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values, by default None.
        Xfit : array-like, optional
            1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values, by default None.
        weight : array-like, optional
            Weights used for fitting, by default None.
            
        Returns
        -------
        float
            Target metric value.
        """        

        if metric_name.lower() == 'intdiff':
            if len(X.shape) == 1:
                metric = np.trapz(np.abs(y-yfit),x=X[:,0])
            else:
                Gfracs, indices = np.unique(X[:,1], return_index=True)
                Gfracs = Gfracs[np.argsort(indices)] # unsure the order of the Gfracs is the same as they are in X 
                
                metric = 0
                for Gfrac in Gfracs:
                    Jmin = min(np.min(y[X[:,1]==Gfrac]),np.min(yfit[X[:,1]==Gfrac]))
                    Jmax = max(np.max(y[X[:,1]==Gfrac]),np.max(yfit[X[:,1]==Gfrac]))
                    Vmin = min(np.min(X[X[:,1]==Gfrac,0]),np.min(X[X[:,1]==Gfrac,0]))
                    Vmax = max(np.max(X[X[:,1]==Gfrac,0]),np.max(X[X[:,1]==Gfrac,0]))
                    metric += np.trapz(np.abs(y[X[:,1]==Gfrac]-yfit[X[:,1]==Gfrac]),x=X[X[:,1]==Gfrac,0]) / ((Jmax-Jmin)*(Vmax-Vmin))
            return metric
        else:
            return  calc_metric(y,yfit,sample_weight=weight,metric_name=metric_name)
    

    def run_Ax(self, parameters):
        """Function to run the simulation with the parameters and return the target metric value for Ax optimization

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.

        Returns
        -------
        dict
            Dictionary with the target metric value and any tracking metrics.
        """  

        df = self.run_JV(parameters)
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
        
        # First loop: calculate main metrics for each exp_format
        for i in range(len(self.exp_format)):
            Xfit, yfit = self.reformat_JV_data(df, self.X[i], self.exp_format[i])
            
            # Apply data transformation based on transforms
            if self.transforms == 'linear':
                metric_value = self.target_metric(
                    self.y[i],
                    yfit,
                    self.metric[i],
                    self.X[i],
                    Xfit,
                    weight=self.weight[i]
                )
            else:
                y_true_transformed, y_pred_transformed = transform_data(
                    self.y[i], 
                    yfit, 
                    X=self.X[i],
                    X_pred=Xfit,
                    transforms=self.transforms,
                    do_G_frac_transform=self.do_G_frac_transform
                )
                
                # Calculate metric with transformed data
                metric_value = calc_metric(
                    y_true_transformed, 
                    y_pred_transformed, 
                    sample_weight=self.weight[i], 
                    metric_name=self.metric[i]
                )
            
            dum_dict[self.all_agent_metrics[i]] = loss_function(metric_value, loss=self.loss[i])
        
        # Second loop: calculate all tracking metrics
        if self.tracking_metric is not None:
            for j in range(len(self.all_agent_tracking_metrics)):
                exp_fmt = self.tracking_exp_format[j]
                metric_name = self.tracking_metric[j]
                loss_type = self.tracking_loss[j]
                
                Xfit, yfit = self.reformat_JV_data(df, self.tracking_X[j], exp_fmt)
                
                # Apply data transformation based on transforms
                if self.transforms == 'linear':
                    metric_value = self.target_metric(
                        self.tracking_y[j],
                        yfit,
                        metric_name,
                        self.tracking_X[j],
                        Xfit,
                        weight=self.tracking_weight[j]
                    )
                else:
                    # Transform data for each format
                    y_true_transformed, y_pred_transformed = transform_data(
                        self.tracking_y[j], 
                        yfit, 
                        X=self.tracking_X[j],
                        X_pred=Xfit,
                        transforms=self.transforms,
                        do_G_frac_transform=self.do_G_frac_transform
                    )
                    
                    # Calculate metric with transformed data
                    metric_value = calc_metric(
                        y_true_transformed, 
                        y_pred_transformed, 
                        sample_weight=self.tracking_weight[j], 
                        metric_name=metric_name
                    )

                dum_dict[self.all_agent_tracking_metrics[j]] = loss_function(metric_value, loss=loss_type)

        return dum_dict
    
    def run_JV(self, parameters):
        """Run the simulation with the parameters and return the simulated values

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.

        Returns
        -------
        dataframe
            Dataframe with the simulated JV data.
        """    

        parallel = self.kwargs.get('parallel', False)
        max_jobs = self.kwargs.get('max_jobs', 1)
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

        # get Gfracs from X
        # check if X is 1D or 2D
        if len(self.X[0].shape) == 1:
            Gfracs = None
            Gfracs_eff = None
        else:
            Gfracs, Gfracs_eff = [], [] # list of Gfracs (real values) and the effective Gfracs (where Gfrac values are corrected by a factor G_eff)
            got_gfrac_none = False
            for xx in self.X:
                if len(xx.shape) == 1:
                    Gfracs = None
                    got_gfrac_none = True
                else:
                    if got_gfrac_none:
                        raise ValueError('all X elements should have the same shape')

                    Gfrac = xx[:,1]
                    for g in Gfrac:
                        if g not in Gfracs:
                            Gfracs.append(g)

            Gfracs = np.asarray(Gfracs)
            if parameters.get('G_eff', None) is not None:
                G_eff = parameters['G_eff']
                Gfracs_eff = Gfracs * G_eff  
            else:
                G_eff = 1.0
                Gfracs_eff = Gfracs              
        
        # prepare the cmd_pars for the simulation
        clean_pars = self.get_SIMsalabim_clean_cmd_pars(parameters)

        # Run the JV simulation
        UUID = self.kwargs.get('UUID',str(uuid.uuid4()))
        
        ret, mess = run_SS_JV(self.simulation_setup, self.session_path, JV_file_name = 'JV.dat', G_fracs = Gfracs_eff, UUID=UUID, cmd_pars=clean_pars, parallel = parallel, max_jobs = max_jobs)
        
        if type(ret) == int:
            if not (ret == 0  or ret == 95) :
                print('Error in running SIMsalabim: '+mess)
                return np.nan
        elif isinstance(ret, subprocess.CompletedProcess):
            if not(ret.returncode == 0 or ret.returncode == 95):
                print('Error in running SIMsalabim: '+mess)
                return np.nan
        else:
            if not all([(res == 0 or res == 95) for res in ret]):
                print('Error in running SIMsalabim: \n')
                for i in range(len(ret)):
                    print(mess[i])
                return np.nan

        if Gfracs is None:
            try:
                df = pd.read_csv(os.path.join(self.session_path, 'JV_'+UUID+'.dat'), sep=r'\s+')
                # delete the file if it exists
                if os.path.exists(os.path.join(self.session_path, 'JV_'+UUID+'.dat')):
                    os.remove(os.path.join(self.session_path, 'JV_'+UUID+'.dat'))
                # same for log
                if os.path.exists(os.path.join(self.session_path, 'log_'+UUID+'.txt')):
                    os.remove(os.path.join(self.session_path, 'log_'+UUID+'.txt'))
                # and scPars
                if os.path.exists(os.path.join(self.session_path, 'scPars_'+UUID+'.txt')):
                    os.remove(os.path.join(self.session_path, 'scPars_'+UUID+'.txt'))

                return df
            except:
                print('No JV data found for UUID '+UUID + ' and cmd_pars '+str(clean_pars))
                return np.nan
        else:
            # make a dummy dataframe and append the dataframes for each Gfrac with a new column for Gfrac
            for _, Gfrac in enumerate(Gfracs_eff):
                try:
                    df = pd.read_csv(os.path.join(self.session_path, 'JV_Gfrac_'+str(Gfrac)+'_'+UUID+'.dat'), sep=r'\s+')
                    df['Gfrac'] = Gfracs[_]  * np.ones_like(df['Vext'].values)
                    if _ == 0:
                        df_all = df
                    else:
                        # concatenate the dataframes
                        df_all = pd.concat([df_all,df],ignore_index=True)
                    # delete the file if it exists
                    if os.path.exists(os.path.join(self.session_path, 'JV_Gfrac_'+str(Gfrac)+'_'+UUID+'.dat')):
                        os.remove(os.path.join(self.session_path, 'JV_Gfrac_'+str(Gfrac)+'_'+UUID+'.dat'))
                    # same for log
                    if os.path.exists(os.path.join(self.session_path, 'log_Gfrac_'+str(Gfrac)+'_'+UUID+'.txt')):
                        os.remove(os.path.join(self.session_path, 'log_Gfrac_'+str(Gfrac)+'_'+UUID+'.txt'))
                    # and scPars
                    if os.path.exists(os.path.join(self.session_path, 'scPars_Gfrac_'+str(Gfrac)+'_'+UUID+'.txt')):
                        os.remove(os.path.join(self.session_path, 'scPars_Gfrac_'+str(Gfrac)+'_'+UUID+'.txt'))

                except Exception as e:
                    print('No JV data found for UUID '+UUID + ' and cmd_pars '+str(clean_pars)+ ' and Gfrac '+str(Gfrac))
                    # print(e)
                    return np.nan

            #reset the index
            # df_all = df_all.reset_index(drop=True)
            # delete the files

            return df_all
    
    def run(self, parameters, X=None, exp_format = 'JV'):
        """Run the simulation with the parameters and return an array with the simulated values in the format specified by exp_format (default is 'JV')

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.
        X : array-like, optional
            1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values, it must match the X values used for the specified exp_format, by default None
        exp_format : str, optional
            Format of the experimental data, by default 'JV'

        Returns
        -------
        array-like
            1-D array with the simulated current values.
        """        

        # run the simulation
        df = self.run_JV(parameters)
        if df is np.nan or len(df) == 0:
            return np.nan

        if X is None:
            X = self.X[0]
        if 'T' in parameters.keys():
            self.T = parameters['T']
        # reformat the data
        Xfit, yfit = self.reformat_JV_data(df, X, exp_format)

        return yfit

    def reformat_JV_data(self, df, X, exp_format='JV'):
        """ Reformat the data depending on the exp_format and X values
        Also interpolates the data if the simulation did not return the same points as the experimental data (i.e. if some points did not converge)

        Parameters
        ----------
        df : dataframe
            Dataframe with the JV data from run_JV function.
        X : array-like
            1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values.
        exp_format : str, optional
            Format of the experimental data, by default 'JV'

        Returns
        -------
        tuple
            Tuple with the reformatted Xfit and yfit values.

        Raises
        ------
        ValueError
            If the exp_format is not 'JV'
        """        
        
        Xfit, yfit = [], []
        do_interp = True
        
        if exp_format == 'JV':
            #  check if Gfrac in df 
            if 'Gfrac' in df.columns:
                Gfracs, indices = np.unique(X[:,1], return_index=True)
                Gfracs = Gfracs[np.argsort(indices)] # unsure the order of the Gfracs is the same as they are in X 

                for Gfrac in Gfracs:
                    df_dum = df[df['Gfrac'] == Gfrac]
                    Vext = np.asarray(df_dum['Vext'].values)
                    Jext = np.asarray(df_dum['Jext'].values)
                    G = np.ones_like(Vext)*Gfrac

                    # check if all points from X[:,0] and data['Vext'].values are the same
                    do_interp = True
                    if len(X[X[:,1]==Gfrac,0]) == len(Vext) :
                        if np.allclose(X[X[:,1]==Gfrac,0], Vext):
                            do_interp = False

                    if do_interp:
                        # Do interpolation in case SIMsalabim did not return the same number of points 
                        # if Vext[0] >
                        try:
                            tck = interpolate.splrep(Vext, Jext, s=0)
                            if len(Xfit) == 0:
                                Xfit = np.vstack((Vext,G)).T
                                yfit = interpolate.splev(X[X[:,1]==Gfrac,0], tck, der=0, ext=0)
                            else:
                                Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                                yfit = np.hstack((yfit,interpolate.splev(X[X[:,1]==Gfrac,0], tck, der=0, ext=0)))
                            
                        except Exception as e:
                            # if min(X[X[:,1]==Gfrac,0])- 0.025 < min(Vext):
                            #     # add a point at the beginning of the JV curve
                            #     Vext
                            f = interpolate.interp1d(Vext, Jext, fill_value='extrapolate', kind='linear')
                            if len(Xfit) == 0:
                                Xfit = np.vstack((Vext,G)).T
                                yfit = f(X[X[:,1]==Gfrac,0])
                            else:
                                Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                                yfit = np.hstack((yfit,f(X[X[:,1]==Gfrac,0])))
                    else:
                        if len(Xfit) == 0:
                            Xfit = np.vstack((Vext,G)).T
                            yfit = Jext
                        else:
                            Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                            yfit = np.hstack((yfit,Jext))
            else:
                Vext = np.asarray(df['Vext'].values)
                Jext = np.asarray(df['Jext'].values)
                G = np.ones_like(Vext)

                # check if all points from X[:,0] and data['Vext'].values are the same
                do_interp = True
                if len(X) == len(Vext) :
                    if np.allclose(X[:,0], Vext):
                        do_interp = False

                if do_interp:
                    # Do interpolation in case SIMsalabim did not return the same number of points 
                    try:
                        tck = interpolate.splrep(Vext, Jext, s=0)
                        yfit = interpolate.splev(X, tck, der=0, ext=0)
                    except:
                        # if min(X)- 0.025 < min(Vext):
                        #     # add a point at the beginning of the JV curve
                        #     df = df.append({'Vext':min(X),'Jext':df['Jext'].iloc[0]},ignore_index=True)
                        #     df = df.sort_values(by=['Vext'])
                        f = interpolate.interp1d(df['Vext'], df['Jext'], fill_value='extrapolate', kind='linear')
                        yfit = f(X)
                else:
                    Xfit = X
                    yfit = Jext
        elif re.match(r'^K_QFLSL-?\d+$', exp_format):
            # the format below is to match the Voltage vs QFLS as described in:
            # Grabowski, D., Liu, Z., Schöpe, G., Rau, U. and Kirchartz, T. (2022)
            # Fill Factor Losses and Deviations from the Superposition Principle in Lead Halide Perovskite Solar Cells. Sol. RRL, 6: 2200507. 
            # https://doi.org/10.1002/solr.202200507
            if 'Gfrac' in df.columns:
                Gfracs, indices = np.unique(X[:,1], return_index=True)
                Gfracs = Gfracs[np.argsort(indices)] # unsure the order of the Gfracs is the same as they are in X 
                # get layer number after K_QFLSL
                m = re.match(r'^K_QFLSL(-?)(\d+)$', exp_format)
                sign = m.group(1)   # '' or '-'
                value = int(m.group(2))
                for Gfrac in Gfracs:
                    df_dum = copy.deepcopy(df[df['Gfrac'] == Gfrac])
                    # reset index
                    df_dum = df_dum.reset_index(drop=True)
                    Vext = np.asarray(df_dum['Vext'].values)
                    Jext = np.asarray(df_dum['Jext'].values)
                    G = np.ones_like(Vext)*Gfrac
                    if Vext[0] > Vext[-1]:
                        Voc_dum = np.interp(0, Jext[::-1], Vext[::-1])
                    else:
                        Voc_dum = np.interp(0, Jext, Vext)
                    idx_Voc = (np.abs(df_dum['Vext']-Voc_dum)).idxmin()
                    JdirOC = df_dum['JdirL'+sign+str(value)].iloc[idx_Voc]
                    kb = constants.value(u'Boltzmann constant in eV/K')
                    # check if self as T attribute
                    if hasattr(self, 'T'):
                        T = self.T
                    else:
                        T = float(self.SIMsalabim_params['setup']['T'])
                    QFLS = kb*T* np.log(df_dum['JdirL'+sign+str(value)].values / JdirOC) + Voc_dum

                    # check if all points from X[:,0] and data['Vext'].values are the same
                    do_interp = True
                    if len(X[X[:,1]==Gfrac,0]) == len(Vext) :
                        if np.allclose(X[X[:,1]==Gfrac,0], Vext):
                            do_interp = False
                    if do_interp:
                        # Do interpolation in case SIMsalabim did not return the same number of points 
                        # if Vext[0] >
                        try:
                            tck = interpolate.splrep(Vext, QFLS, s=0)
                            if len(Xfit) == 0:
                                Xfit = np.vstack((Vext,G)).T
                                yfit = interpolate.splev(X[X[:,1]==Gfrac,0], tck, der=0, ext=0)
                            else:
                                Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                                yfit = np.hstack((yfit,interpolate.splev(X[X[:,1]==Gfrac,0], tck, der=0, ext=0)))
                            
                        except Exception as e:
                            # if min(X[X[:,1]==Gfrac,0])- 0.025 < min(Vext):
                            #     # add a point at the beginning of the JV curve
                            #     Vext
                            f = interpolate.interp1d(Vext, QFLS, fill_value='extrapolate', kind='linear')
                            if len(Xfit) == 0:
                                Xfit = np.vstack((Vext,G)).T
                                yfit = f(X[X[:,1]==Gfrac,0])
                            else:
                                Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                                yfit = np.hstack((yfit,f(X[X[:,1]==Gfrac,0])))
                    else:
                        if len(Xfit) == 0:
                            Xfit = np.vstack((Vext,G)).T
                            yfit = QFLS
                        else:
                            Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                            yfit = np.hstack((yfit,QFLS))
            else:
                Vext = np.asarray(df['Vext'].values)
                Jext = np.asarray(df['Jext'].values)
                G = np.ones_like(Vext)
                Voc_dum = np.interp(0, Jext, Vext)
                idx_Voc = (np.abs(df['Vext']-Voc_dum)).idxmin()
                JdirOC = df['JdirL'+sign+str(value)].iloc[idx_Voc]
                kb = constants.value(u'Boltzmann constant in eV/K')
                # check if self as T attribute
                if hasattr(self, 'T'):
                    T = self.T
                else:
                    T = self.dev_par['T']
                QFLS = kb*T* np.log(df['JdirL'+sign+str(value)].values / JdirOC) + Voc_dum

                # check if all points from X[:,0] and data['Vext'].values are the same
                do_interp = True
                if len(X) == len(Vext) :
                    if np.allclose(X[:,0], Vext):
                        do_interp = False

                if do_interp:
                    # Do interpolation in case SIMsalabim did not return the same number of points 
                    try:
                        tck = interpolate.splrep(Vext, QFLS, s=0)
                        yfit = interpolate.splev(X, tck, der=0, ext=0)
                    except:
                        # if min(X)- 0.025 < min(Vext):
                        #     # add a point at the beginning of the JV curve
                        #     df = df.append({'Vext':min(X),'Jext':df['Jext'].iloc[0]},ignore_index=True)
                        #     df = df.sort_values(by=['Vext'])
                        f = interpolate.interp1d(df['Vext'], QFLS, fill_value='extrapolate', kind='linear')
                        yfit = f(X)
                else:
                    Xfit = X
                    yfit = QFLS

        elif exp_format in df.columns:
            if 'Gfrac' in df.columns:
                Gfracs, indices = np.unique(X[:,1], return_index=True)
                Gfracs = Gfracs[np.argsort(indices)] # unsure the order of the Gfracs is the same as they are in X 

                for Gfrac in Gfracs:
                    df_dum = df[df['Gfrac'] == Gfrac]
                    Vext = np.asarray(df_dum['Vext'].values)
                    exp = np.asarray(df_dum[exp_format].values)
                    G = np.ones_like(Vext)*Gfrac

                    # check if all points from X[:,0] and data['Vext'].values are the same
                    do_interp = True
                    if len(X[X[:,1]==Gfrac,0]) == len(Vext) :
                        if np.allclose(X[X[:,1]==Gfrac,0], Vext):
                            do_interp = False

                    if do_interp:
                        # Do interpolation in case SIMsalabim did not return the same number of points 
                        # if Vext[0] >
                        try:
                            tck = interpolate.splrep(Vext, exp, s=0)
                            if len(Xfit) == 0:
                                Xfit = np.vstack((Vext,G)).T
                                yfit = interpolate.splev(X[X[:,1]==Gfrac,0], tck, der=0, ext=0)
                            else:
                                Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                                yfit = np.hstack((yfit,interpolate.splev(X[X[:,1]==Gfrac,0], tck, der=0, ext=0)))
                            
                        except Exception as e:
                            # if min(X[X[:,1]==Gfrac,0])- 0.025 < min(Vext):
                            #     # add a point at the beginning of the JV curve
                            #     Vext
                            f = interpolate.interp1d(Vext, exp, fill_value='extrapolate', kind='linear')
                            if len(Xfit) == 0:
                                Xfit = np.vstack((Vext,G)).T
                                yfit = f(X[X[:,1]==Gfrac,0])
                            else:
                                Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                                yfit = np.hstack((yfit,f(X[X[:,1]==Gfrac,0])))
                    else:
                        if len(Xfit) == 0:
                            Xfit = np.vstack((Vext,G)).T
                            yfit = exp
                        else:
                            Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                            yfit = np.hstack((yfit,exp))
            else:
                Vext = np.asarray(df['Vext'].values)
                exp = np.asarray(df[exp_format].values)
                G = np.ones_like(Vext)

                # check if all points from X[:,0] and data['Vext'].values are the same
                do_interp = True
                if len(X) == len(Vext) :
                    if np.allclose(X[:,0], Vext):
                        do_interp = False

                if do_interp:
                    # Do interpolation in case SIMsalabim did not return the same number of points 
                    try:
                        tck = interpolate.splrep(Vext, exp, s=0)
                        yfit = interpolate.splev(X, tck, der=0, ext=0)
                    except:
                        # if min(X)- 0.025 < min(Vext):
                        #     # add a point at the beginning of the JV curve
                        #     df = df.append({'Vext':min(X),'Jext':df['Jext'].iloc[0]},ignore_index=True)
                        #     df = df.sort_values(by=['Vext'])
                        f = interpolate.interp1d(df['Vext'], df[exp_format], fill_value='extrapolate', kind='linear')
                        yfit = f(X)
                else:
                    Xfit = X
                    yfit = exp
        else:
            raise ValueError('Invalid exp_format: '+exp_format)
        
        return Xfit, yfit