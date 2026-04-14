"""CVAgent class for steady-state CV simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy
from scipy import interpolate

from optimpv import *
from optimpv.general.general import calc_metric, loss_function, transform_data
from optimpv.models.DDfits.SIMsalabimAgent import SIMsalabimAgent
from pySIMsalabim import *
from pySIMsalabim.experiments.CV import *

######### Agent Definition #######################################################################
class CVAgent(SIMsalabimAgent):  
    """CVAgent class for Capacitance-Voltage simulations with SIMsalabim

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
    V_step : float, optional
        Voltage difference, determines at which voltages the capacitance is determined, by default 0.1.
    freq : float, optional
        Frequency of the CV measurement [Hz], by default 1e4.
    G_frac : float, optional
        Fractional light intensity, by default 1.
    del_V : float, optional
        Voltage step for the impedance simulation, by default 0.01.
    simulation_setup : str, optional
        Path to the simulation setup file, if None then use the default file 'simulation_setup.txt'in the session_path directory, by default None.
    exp_format : str or list of str, optional
        Format of the CV data, possible values are: 'CV', 'MottSchottky', by default 'CV'.
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
        Name of the agent, by default 'CV'.
    **kwargs : dict
        Additional keyword arguments.
    """   
    def __init__(self, params, X, y, session_path, Vmin=0.5, Vmax=1.0, V_step=0.1,freq=1e4, G_frac=1, del_V=0.01, 
                 simulation_setup=None, exp_format='CV', metric='mse', loss='linear', threshold=100, minimize=True, 
                 yerr=None, weight=None, tracking_metric=None, tracking_loss=None, tracking_exp_format=None, 
                 tracking_X=None, tracking_y=None, tracking_weight=None, transforms='linear', tracking_transforms = 'linear', name='CV', **kwargs):    
        super().__init__(params, X, y, session_path, simulation_setup, exp_format, metric, loss, threshold, minimize, yerr, weight, tracking_metric, tracking_loss, tracking_exp_format, tracking_X, tracking_y, tracking_weight, transforms, tracking_transforms, name, **kwargs)

        # CVAgent specific parameters
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.V_step = V_step
        self.freq = freq
        self.G_frac = G_frac
        self.del_V = del_V

       

    def validate_exp_format(self, exp_format):
        """Validate the exp_format parameter to ensure it is in the correct format for CVAgent

        Parameters
        ----------
        exp_format : str
            Format of the experimental data, must be one of the formats in self.exp_format.

        Raises
        ------
        ValueError
            If the exp_format is not valid
        """         
        if exp_format.lower() not in ['cv','mottschottky']:
            raise ValueError(f'{exp_format} is an invalid CV format. Possible values are: CV and MottSchottky.')
        
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
        df = self.run_CV_simulation(parameters)

        return self._run_Ax(df,self.reformat_CV_data)
    
    def run_CV_simulation(self, parameters):
        """Run the simulation with the parameters and return the simulated values

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.

        Returns
        -------
        dataframe
            Dataframe with the simulated CV values.
        """    

        parallel = self.kwargs.get('parallel', False)
        max_jobs = self.kwargs.get('max_jobs', 1)
        # output_file = self.kwargs.get('output_file', 'CapVol.dat')

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
        
        # Run the CV simulation
        UUID = self.kwargs.get('UUID',str(uuid.uuid4()))

        # remove UUID and output_file and cmd_pars from kwargs
        dummy_kwargs = copy.deepcopy(self.kwargs)
        if 'UUID' in dummy_kwargs:
            dummy_kwargs.pop('UUID')
        if 'output_file' in dummy_kwargs:
            dummy_kwargs.pop('output_file')
        if 'cmd_pars' in dummy_kwargs:
            dummy_kwargs.pop('cmd_pars')

        ret, mess = run_CV_simu(self.simulation_setup, self.session_path, self.freq, self.Vmin, self.Vmax, self.V_step, G_frac=self.G_frac, del_V=self.del_V,  run_mode=False, output_file = 'CapVol.dat', UUID=UUID, cmd_pars=clean_pars, **dummy_kwargs)
        
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
            df = pd.read_csv(os.path.join(self.session_path, 'CapVol_'+UUID+'.dat'), sep=r'\s+')
        except:
            print('No CV data found for UUID '+UUID + ' and cmd_pars '+str(cmd_pars))
            return np.nan

        return df

    def run(self, parameters,X=None,exp_format='CV'):
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

        df = self.run_CV_simulation(parameters)

        if df is np.nan or len(df) == 0:
            return np.nan

        if X is None:
            X = self.X[0]

        Xfit, yfit = self.reformat_CV_data(df, X, exp_format)

        return yfit


    def reformat_CV_data(self,df,X,exp_format='CV'):
        """ Reformat the data depending on the exp_format and X values
        Also interpolates the data if the simulation did not return the same points as the experimental data (i.e. if some points did not converge)

        Parameters
        ----------
        df : dataframe
            Dataframe with the CV dara from run_CV_simulation function.
        X : array-like, optional
            1-D array containing the x axis values, by default None.
        exp_format : str, optional
            Format of the experimental data, by default 'CV'.

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
        if exp_format.lower() == 'cv':
            
            if len(X) == len(df['V'].values):
                if np.allclose(X, np.asarray(df['V'].values)):
                    do_interp = False
            
            if do_interp:

                # Do interpolation in case SIMsalabim did not return the same number of points as the experimental data
                try:
                    tck = interpolate.splrep(df['V'], df['C'].values, s=0)
                    yfit = interpolate.splev(X, tck, der=0)
                except:
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    f = interpolate.interp1d(df['V'], df['C'].values, kind='linear', fill_value='extrapolate')
                    yfit = f(X)
            else:
                Xfit = X
                yfit = np.asarray(df['C'].values)
        
        elif exp_format.lower() == 'mottschottky':

            if len(X) == len(df['V'].values):
                if np.allclose(X, np.asarray(df['V'].values)):
                    do_interp = False
            
            if do_interp:

                # Do interpolation in case SIMsalabim did not return the same number of points as the experimental data
                try:
                    tck = interpolate.splrep(df['V'], 1/df['C'].values**2, s=0)
                    yfit = interpolate.splev(X, tck, der=0)
                except:
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    f = interpolate.interp1d(df['V'], 1/df['C'].values**2, kind='linear', fill_value='extrapolate')
                    yfit = f(X)
            else:
                Xfit = X
                yfit = np.asarray(1/df['C'].values**2)


       
        else:
            raise ValueError('Invalid CV format. Possible values are: CV.')

        return Xfit, yfit