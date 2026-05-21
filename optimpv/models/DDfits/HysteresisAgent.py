"""HysteresisAgent class for transient hysteresis JV simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy,warnings
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
                 transforms='linear', tracking_transforms='linear',
                 name='Hyst', **kwargs):    

        super().__init__(params, X, y, session_path, simulation_setup, exp_format, metric, loss, threshold, minimize, yerr, weight, tracking_metric, tracking_loss, tracking_exp_format, tracking_X, tracking_y, tracking_weight, transforms, tracking_transforms, name, **kwargs)

        # HysteresisAgent specific parameters
        self.scan_speed = scan_speed
        self.direction = direction
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.steps = steps
        self.G_frac = G_frac

    def validate_exp_format(self, exp_format):
        """Validate the exp_format parameter to ensure it is in the correct format for HysteresisAgent

        Parameters
        ----------
        exp_format : str
            Format of the experimental data, must be one of the allowed formats for hysteresis data.

        Raises
        ------
        ValueError
            If the exp_format is not valid or not found in self.exp_format.
        """    
        if exp_format not in ['JV']:
            raise ValueError(f'{exp_format} is an invalid hysteresis format. Possible values are: JV.')

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
        return self._run_Ax(df,self.reformat_hysteresis_data)

    
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


        if parameters.get('G_eff', None) is not None:
            G_eff = parameters['G_eff']
            Gfracs_eff = self.G_frac * G_eff  
        elif 'G_eff' in self.pnames:
            idx_G_eff = self.pnames.index('G_eff')
            G_eff = self.params[idx_G_eff].value
            Gfracs_eff = self.G_frac * G_eff
        else:
            G_eff = 1.0
            Gfracs_eff = self.G_frac

        ret, mess, rms = Hysteresis_JV(self.simulation_setup, self.session_path, 0, scan_speed=self.scan_speed, direction=self.direction, G_frac=Gfracs_eff, Vmin=self.Vmin, Vmax=self.Vmax, steps=self.steps, UUID=UUID, cmd_pars=clean_pars, tj_name= 'tj.dat', **dummy_kwargs)
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
        if df is np.nan or len(df) == 0:
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
                
                t_exp = np.zeros_like(X)
                t_exp[0] = np.abs(X[0] - Vext[0])/self.scan_speed
                for i in range(1, len(X)):
                    t_exp[i] = t_exp[i-1] + np.abs(X[i] - X[i-1])/self.scan_speed
                # Do interpolation in case SIMsalabim did not return the same number of points as the experimental data
                # we do this with the time axis and not the voltage axis since the time axis is strictly increasing, this avoids problems with the spline interpolation and avoid spliting the data in two parts
                
                # Split the data into forward and backward scans
                if self.direction == 1: # Forward-backward
                    split_idx_X = np.argmax(X) + 1
                    split_idx_sim = np.argmax(Vext) + 1
                else: # Backward-forward
                    split_idx_X = np.argmin(X) + 1
                    split_idx_sim = np.argmin(Vext) + 1

                X_fwd, X_bwd = np.split(X, [split_idx_X])
                
                V_sim_fwd, V_sim_bwd = np.split(Vext, [split_idx_sim])
                J_sim_fwd, J_sim_bwd = np.split(df['Jext'].values, [split_idx_sim])

                yfit_fwd, yfit_bwd = np.array([]), np.array([])

                # Interpolate forward scan
                try:
                    # Ensure V_sim_fwd is monotonic before spline interpolation
                    sort_indices = np.argsort(V_sim_fwd)
                    tck_fwd = interpolate.splrep(V_sim_fwd[sort_indices], J_sim_fwd[sort_indices], s=0)
                    yfit_fwd = interpolate.splev(X_fwd, tck_fwd, der=0)
                except Exception:
                    warnings.warn('Spline interpolation failed for forward scan, using linear interpolation', UserWarning)
                    f_fwd = interpolate.interp1d(V_sim_fwd, J_sim_fwd, kind='linear', fill_value='extrapolate', bounds_error=False)
                    yfit_fwd = f_fwd(X_fwd)

                # Interpolate backward scan
                try:
                    # Ensure V_sim_bwd is monotonic before spline interpolation
                    sort_indices = np.argsort(V_sim_bwd)
                    tck_bwd = interpolate.splrep(V_sim_bwd[sort_indices], J_sim_bwd[sort_indices], s=0)
                    yfit_bwd = interpolate.splev(X_bwd, tck_bwd, der=0)
                except Exception:
                    warnings.warn('Spline interpolation failed for backward scan, using linear interpolation', UserWarning)
                    f_bwd = interpolate.interp1d(V_sim_bwd, J_sim_bwd, kind='linear', fill_value='extrapolate', bounds_error=False)
                    yfit_bwd = f_bwd(X_bwd)
                
                yfit = np.concatenate((yfit_fwd, yfit_bwd))
                Xfit = X
            else:
                Xfit = X
                yfit = np.asarray(df['Jext'].values)

       
        else:
            raise ValueError('Invalid hysteresis format. Possible values are: JV.')

        return Xfit, yfit