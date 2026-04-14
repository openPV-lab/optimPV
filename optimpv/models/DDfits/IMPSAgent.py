"""IMPSAgent class for steady-state IMPS simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy, time, warnings
from scipy import interpolate

from optimpv import *
from optimpv.general.general import *
from optimpv.models.DDfits.SIMsalabimAgent import SIMsalabimAgent
from pySIMsalabim import *
from pySIMsalabim.experiments.imps import *

######### Agent Definition #######################################################################
class IMPSAgent(SIMsalabimAgent):  
    """IMPSAgent class for IMPS simulations with SIMsalabim

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
    f_min : float
        Minimum frequency for the IMPS simulation in Hz.
    f_max : float
        Maximum frequency for the IMPS simulation in Hz.
    f_steps : float, optional
        Number of frequency steps for the IMPS simulation (log spaced), by default 30.
    V : float, optional
        Voltage value for the simulation, by default 0.
    G_frac : float, optional
        Fractional light intensity, by default 1.
    GStep : float, optional
        Applied generation rate increase at t=0, by default 0.05.
    simulation_setup : str, optional
        Path to the simulation setup file, if None then use the default file 'simulation_setup.txt'in the session_path directory, by default None.
    exp_format : str, optional
        Format of the IMPS data, possible values are: 'ReY', 'ImY', 'ColeCole', by default 'ImY'.
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
        Name of the agent, by default 'IMPS'.
    **kwargs : dict
        Additional keyword arguments.
    """   
    def __init__(self, params, X, y, session_path, f_min, f_max, f_steps=30, V=0, G_frac=1, GStep=0.05, 
                 simulation_setup=None, exp_format='ImY', metric='mse', loss='linear', threshold=100, minimize=True, 
                 yerr=None, weight=None, tracking_metric=None, tracking_loss=None, tracking_exp_format=None, 
                 tracking_X=None, tracking_y=None, tracking_weight=None, transforms='linear', tracking_transforms = 'linear', name='IMPS', **kwargs):    

        super().__init__(params, X, y, session_path, simulation_setup, exp_format, metric, loss, threshold, minimize, yerr, weight, tracking_metric, tracking_loss, tracking_exp_format, tracking_X, tracking_y, tracking_weight, transforms, tracking_transforms, name, **kwargs)

        # IMPSAgent specific parameters
        self.f_min = f_min
        self.f_max = f_max
        self.f_steps = f_steps
        self.V = V
        self.G_frac = G_frac
        self.GStep = GStep

    def validate_exp_format(self, exp_format):
        """Validate the exp_format parameter to ensure it is in the correct format for IMPSAgent

        Parameters
        ----------
        exp_format : str
            Format of the experimental data, must be one of the allowed formats for IMPS data: 'ReY', 'ImY', 'ColeCole'.

        Raises
        ------
        ValueError
            If the exp_format is not valid
        """            
        
        if exp_format.lower() not in ['rey', 'imy', 'colecole']:
            raise ValueError(f'{exp_format} is an invalid IMPS format. Possible values are: ReY, ImY, ColeCole')
        
    def target_metric(self, y, yfit, metric_name, X=None, Xfit=None, weight=None):
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
        if metric_name.lower() == 'mmeud':
            if Xfit is None:
                raise ValueError('Xfit must be specified for the mmed metric')
            return mean_min_euclidean_distance(X,y,Xfit,yfit)
        elif metric_name.lower() == 'dmeud':
            if Xfit is None:
                raise ValueError('Xfit must be specified for the med metric')
            return direct_mean_euclidean_distance(X,y,Xfit,yfit)
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
        df = self.run_IMPS_simulation(parameters)

        return self._run_Ax(df,self.reformat_IMPS_data)
    
    def run_IMPS_simulation(self, parameters):
        """Run the simulation with the parameters and return the simulated values

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.

        Returns
        -------
        dataframe
            Dataframe with the simulated IMPS values.
        """    

        parallel = self.kwargs.get('parallel', False)
        max_jobs = self.kwargs.get('max_jobs', 1)
        # output_file = self.kwargs.get('output_file', 'freqY.dat')

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

        # Run the IMPS simulation
        UUID = self.kwargs.get('UUID',str(uuid.uuid4()))

        # remove UUID and output_file and cmd_pars from kwargs
        dummy_kwargs = copy.deepcopy(self.kwargs)
        if 'UUID' in dummy_kwargs:
            dummy_kwargs.pop('UUID')
        if 'output_file' in dummy_kwargs:
            dummy_kwargs.pop('output_file')
        if 'cmd_pars' in dummy_kwargs:
            dummy_kwargs.pop('cmd_pars')

        ret, mess = run_IMPS_simu(self.simulation_setup, self.session_path, self.f_min, self.f_max, self.f_steps, self.V, self.G_frac, self.GStep,  run_mode=False, output_file = 'freqY.dat', UUID=UUID, cmd_pars=clean_pars, **dummy_kwargs)
        
        if type(ret) == int: 
            if not (ret == 0  or ret == 95):
                print('Error in running SIMsalabim: '+mess)
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
            df = pd.read_csv(os.path.join(self.session_path, 'freqY_'+UUID+'.dat'), sep=r'\s+')
        except:
            print('No IMPS data found for UUID '+UUID + ' and cmd_pars '+str(cmd_pars))
            return np.nan

        return df

    def run(self, parameters,X=None,exp_format='ImY'):
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

        df = self.run_IMPS_simulation(parameters)
        if df is np.nan or len(df) == 0:
            return np.nan

        if X is None:
            X = self.X[0]

        Xfit, yfit = self.reformat_IMPS_data(df, X, exp_format)

        return yfit


    def reformat_IMPS_data(self,df,X,exp_format='IMPS'):
        """ Reformat the data depending on the exp_format and X values
        Also interpolates the data if the simulation did not return the same points as the experimental data (i.e. if some points did not converge)

        Parameters
        ----------
        df : dataframe
            Dataframe with the IMPS dara from run_IMPS_simulation function.
        X : array-like, optional
            1-D array containing the x axis values, by default None.
        exp_format : str, optional
            Format of the experimental data, by default 'IMPS'.

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
        if exp_format.lower() == 'rey':
            
            if len(X) == len(df['freq'].values):
                if np.allclose(X, np.asarray(df['freq'].values)):
                    do_interp = False
            
            if do_interp:

                # Do interpolation in case SIMsalabim did not return the same number of points as the experimental data
                try:
                    tck = interpolate.splrep(df['freq'], df['ReY'].values, s=0)
                    yfit = interpolate.splev(X, tck, der=0)
                except:
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    f = interpolate.interp1d(df['freq'], df['ReY'].values, kind='linear', fill_value='extrapolate')
                    yfit = f(X)
            else:
                Xfit = X
                yfit = np.asarray(df['ReY'].values)

        elif exp_format.lower() == 'imy':
                

                if len(X) == len(df['freq'].values):
                    if np.allclose(X, np.asarray(df['freq'].values)):
                        do_interp = False
                
                if do_interp:
    
                    # Do interpolation in case SIMsalabim did not return the same number of points as the experimental data
                    try:
                        tck = interpolate.splrep(df['freq'], df['ImY'].values, s=0)
                        yfit = interpolate.splev(X, tck, der=0)
                    except:
                        warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                        f = interpolate.interp1d(df['freq'], df['ImY'].values, kind='linear', fill_value='extrapolate')
                        yfit = f(X)
                else:
                    Xfit = X
                    yfit = np.asarray(df['ImY'].values)
        
        elif exp_format.lower() == 'colecole':

            if self.metric.lower() == 'mmeud' or self.metric.lower() == 'dmeud':
                Xfit = np.asarray(df['ReY'].values)
                yfit = np.asarray(df['ImY'].values)
            else:
                raise ValueError('Invalid metric for Cole-Cole analysis. Possible values are: MMEUD, DMEUD. if you want to fit the ReZ and ImZ values, please for a MO analysis using the ReY and ImY exp_format')
            
            if len(X) == len(Xfit):
                if np.allclose(X, Xfit):
                    do_interp = False

            if do_interp:
                freqs = self.kwargs.get('freqs',None)
                if freqs is None:
                    raise ValueError('freqs must be specified for a Cole-Cole analysis in case not all frequencies are returned by SIMsalabim')

                try:
                    # interpolate ReZ
                    dum_freqs = np.asarray(df['freq'].values)
                    dum_Re = np.asarray(df['ReY'].values)
                    dum_Im = np.asarray(df['ImY'].values)
                    # check if the frequencies are in descending order and reverse them if necessary
                    if dum_freqs[0] > dum_freqs[-1]:
                        dum_freqs = dum_freqs[::-1]
                        dum_Re = dum_Re[::-1]
                        dum_Im = dum_Im[::-1]

                    tck = interpolate.splrep(dum_freqs, dum_Re, s=0)
                    yfit = interpolate.splev(freqs, tck, der=0, ext=0)
                    # interpolate ImZ
                    tck = interpolate.splrep(dum_freqs, dum_Im, s=0)
                    yfit2 = interpolate.splev(freqs, tck, der=0, ext=0)

                    Xfit = yfit
                    yfit = yfit2

                except Exception as e:
                    f = interpolate.interp1d(np.asarray(df['freq'].values), np.asarray(df['ReY'].values), fill_value='extrapolate', kind='linear')
                    yfit = f(freqs)
                    f = interpolate.interp1d(np.asarray(df['freq'].values), np.asarray(df['ImY'].values), fill_value='extrapolate', kind='linear')
                    yfit2 = f(freqs)
                    # put ReZ and ImZ in the same array and double the length of Xfit
                    Xfit = yfit
                    yfit = yfit2

                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
        else:
            raise ValueError('Invalid IMPS format. Possible values are: IMPS.')

        return Xfit, yfit