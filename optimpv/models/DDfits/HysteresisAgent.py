"""HysteresisAgent class for transient hysteresis JV simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy
from scipy import interpolate

from optimpv import *
from optimpv.general.general import calc_metric, loss_function, transform_data
from optimpv.models.DDfits.SIMsalabimAgent import SIMsalabimAgent
from pySIMsalabim import *
from pySIMsalabim.experiments.hysteresis import *

######### Agent Definition #######################################################################
class HysteresisAgent(SIMsalabimAgent):  
    """HysteresisAgent class for JV hysteresis simulations with SIMsalabim

    Parameters
    ----------
    params : list of Fitparam() objects
        List of Fitparam() objects.
    X : array-like
        1-D or 2-D array containing the voltage values.
    y : array-like
        1-D array containing the current values.
    session_path : str
        Path to the session directory.
    Vmin : float, optional
        minimum voltage, by default 0.
    Vmax : float, optional
        maximum voltage, by default 1.2.
    scan_speed : float, optional
        Voltage scan speed [V/s], by default 0.1.
    steps : int, optional
        Number of time steps, by default 100.
    direction : integer, optional
        Perform a forward-backward (1) or backward-forward scan (-1), by default 1.
    G_frac : float, optional
        Fractional light intensity, by default 1.
    simulation_setup : str, optional
        Path to the simulation setup file, if None then use the default file 'simulation_setup.txt'in the session_path directory, by default None.
    exp_format : str or list of str, optional
        Format of the hysteresis data, possible values are: 'JV', by default 'JV'.
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
        Name of the agent, by default 'Hyst'.
    **kwargs : dict
        Additional keyword arguments.
    """   
    def __init__(self, params, X, y, session_path, Vmin=0, Vmax=1.2, scan_speed=0.1, steps=100, 
                 direction=1, G_frac=1, simulation_setup=None, exp_format='JV', 
                 metric='mse', loss='linear', threshold=100, minimize=True, 
                 yerr=None, weight=None, tracking_metric=None, tracking_loss=None,
                 tracking_exp_format=None, tracking_X=None, tracking_y=None, tracking_weight=None,
                 transforms='linear', name='Hyst', **kwargs):    

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
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.scan_speed = scan_speed
        self.steps = steps
        self.direction = direction
        self.G_frac = G_frac
        self.name = name
        
        self.exp_format = exp_format
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]

        # check that all elements in exp_format are valid
        for hyst_form in self.exp_format:
            if hyst_form not in ['JV']:
                raise ValueError('{hyst_form} is an invalid hysteresis format. Possible values are: JV.')

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
                if form not in ['JV']:
                    raise ValueError(f'{form} is an invalid tracking_exp_format, must be "JV"')
            
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
        
        self.all_agent_metrics = self.get_all_agent_metric_names()
        self.all_agent_tracking_metrics = self.get_all_agent_tracking_metric_names()
            
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

    def target_metric(self, y, yfit, metric_name, X=None, Xfit=None,weight=None):
        """Calculate the target metric depending on self.metric

        Parameters
        ----------
        y : array-like
            1-D array containing the target values.
        yfit : array-like
            1-D array containing the fitted values.
        metric_name : str
            Metric to evaluate the model, see optimpv.general.calc_metric for options.
        X : array-like, optional
            1-D array containing the x axis values, by default None.
        Xfit : array-like, optional
            1-D array containing the x axis values, by default None.
        weight : array-like, optional
            1-D array containing the weights, by default None.

        Returns
        -------
        float
            Target metric value.
        """        
        
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
            Dictionary with the target metric value.
        """  
        df = self.run_hysteresis_simulation(parameters)
        if (df is np.nan) or (len(df) == 0):
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
            Xfit, yfit = self.reformat_hysteresis_data(df, self.X[i], exp_format=self.exp_format[i])
            
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
                    transforms=self.transforms
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
                
                Xfit, yfit = self.reformat_hysteresis_data(df, self.tracking_X[j], exp_format=exp_fmt)
                
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
                        transforms=self.transforms
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
    
    def run_hysteresis_simulation(self, parameters):
        """Run the simulation with the parameters and return the simulated values

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.

        Returns
        -------
        dataframe
            Dataframe with the simulated hysteresis values.
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

        # prepare the cmd_pars for the simulation
        clean_pars = self.get_SIMsalabim_clean_cmd_pars(parameters)
        
        # Run the JV simulation
        UUID = self.kwargs.get('UUID',str(uuid.uuid4()))

        # remove UUID and output_file and cmd_pars from kwargs
        dummy_kwargs = copy.deepcopy(self.kwargs)
        if 'UUID' in dummy_kwargs:
            dummy_kwargs.pop('UUID')
        if 'output_file' in dummy_kwargs:
            dummy_kwargs.pop('output_file')
        if 'cmd_pars' in dummy_kwargs:
            dummy_kwargs.pop('cmd_pars')

        ret, mess, rms = Hysteresis_JV(self.simulation_setup, self.session_path, 0, scan_speed=self.scan_speed, direction=self.direction, G_frac=self.G_frac, Vmin=self.Vmin, Vmax=self.Vmax, steps=self.steps, UUID=UUID, cmd_pars=clean_pars, tj_name= 'tj.dat', **dummy_kwargs)
        if type(ret) == int:
            if not (ret == 0  or ret == 95):
                # print('Error in running SIMsalabim: '+mess)
                return np.nan
        elif isinstance(ret, subprocess.CompletedProcess):
            
            if not(ret.returncode == 0 or ret.returncode == 95):
                # print('Error in running SIMsalabim: '+mess)
                return np.nan
        else:
            if not all([(res == 0 or res == 95) for res in ret]):
                # print('Error in running SIMsalabim: '+mess)
                return np.nan
        try:
            df = pd.read_csv(os.path.join(self.session_path, 'tj_'+UUID+'.dat'), sep=r'\s+')
        except:
            print('No hysteresis data found for UUID '+UUID + ' and cmd_pars '+str(cmd_pars))
            return np.nan

        return df

    def run(self, parameters,X=None,exp_format='JV'):
        """Run the simulation with the parameters and return an array with the simulated values in the format specified by exp_format (default is 'Cf')

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.
        X : array-like, optional
            1-D array containing the x axis values, by default None.
        exp_format : str, optional
            Format of the experimental data, by default 'Cf'.

        Returns
        -------
        array-like
            1-D array with the simulated current values.
        """     

        df = self.run_hysteresis_simulation(parameters)
        if df is np.nan:
            return np.nan

        if X is None:
            X = self.X[0]

        Xfit, yfit = self.reformat_hysteresis_data(df, X, exp_format)

        return yfit


    def reformat_hysteresis_data(self,df,X,exp_format='JV'):
        """ Reformat the data depending on the exp_format and X values
        Also interpolates the data if the simulation did not return the same points as the experimental data (i.e. if some points did not converge)

        Parameters
        ----------
        df : dataframe
            Dataframe with the hysteresis dara from run_hysteresis_simulation function.
        X : array-like, optional
            1-D array containing the x axis values, by default None.
        exp_format : str, optional
            Format of the experimental data, by default 'JV'.

        Returns
        -------
        tuple
            Tuple with the reformatted Xfit and yfit values.

        Raises
        ------
        ValueError
            If the exp_format is not valid.
        """     
        Xfit,yfit = [],[]
        do_interp = True
        if exp_format == 'JV':
            
            if len(X) == len(df['Vext'].values):
                if np.allclose(X, np.asarray(df['Vext'].values)):
                    do_interp = False

            if do_interp:
                # calcuate time for each voltage step
                t_sim = df['t'].values
                Vext = df['Vext'].values

                t_exp = np.zeros(len(X))
                t_exp[0] = 0
                for i in range(1,len(X)):
                    t_exp[i] = t_exp[i-1] + abs(X[i]-X[i-1])/self.scan_speed

                # Do interpolation in case SIMsalabim did not return the same number of points as the experimental data
                # we do this with the time axis and not the voltage axis since the time axis is strictly increasing, this avoids problems with the spline interpolation and avoid spliting the data in two parts
                try:
                    tck = interpolate.splrep(t_sim, df['Jext'].values, s=0)
                    yfit = interpolate.splev(t_exp, tck, der=0)
                except:
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    f = interpolate.interp1d(t_sim, df['Jext'].values, kind='linear', fill_value='extrapolate')
                    yfit = f(t_exp)
            else:
                Xfit = X
                yfit = np.asarray(df['Jext'].values)

       
        else:
            raise ValueError('Invalid hysteresis format. Possible values are: JV.')

        return Xfit, yfit
