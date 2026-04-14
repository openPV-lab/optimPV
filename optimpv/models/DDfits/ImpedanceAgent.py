"""ImpedanceAgent class for impedance simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy, time, warnings
from scipy import interpolate

from optimpv import *
from optimpv.general.general import *
from optimpv.models.DDfits.SIMsalabimAgent import SIMsalabimAgent
from pySIMsalabim import *
from pySIMsalabim.experiments.impedance import *

######### Agent Definition #######################################################################
class ImpedanceAgent(SIMsalabimAgent):  
    """ImpedanceAgent class for impedance simulations with SIMsalabim

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
    f_min : float
        Minimum frequency for the impedance simulation.
    f_max : float
        Maximum frequency for the impedance simulation.
    f_steps : int
        Number of frequency steps for the impedance simulation.
    V_0 : float
        Initial voltage for the impedance simulation.
    G_frac : float, optional
        Fractional light intensity, by default 1.
    del_V : float, optional
        Voltage step for the impedance simulation, by default 0.01.
    simulation_setup : str, optional
        Path to the simulation setup file, if None then use the default file 'simulation_setup.txt'in the session_path directory, by default None.
    exp_format : str or list of str, optional
        Format of the impedance data, possible values are: 'Cf', 'Gf', 'Nyquist', 'BodeImZ', 'BodeReZ', 'Bode', by default 'Cf'.
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
        Name of the agent, by default 'Imp'.
    **kwargs : dict
        Additional keyword arguments.
    """   
    def __init__(self, params, X, y, session_path, f_min, f_max, f_steps, V_0, G_frac = 1, del_V = 0.01,
                 simulation_setup = None, exp_format = ['Cf'], metric = ['mse'], loss = ['linear'], 
                 threshold = [100], minimize = [True], yerr = None, weight = None, 
                 tracking_metric = None, tracking_loss = None, tracking_exp_format = None, 
                 tracking_X = None, tracking_y = None, tracking_weight = None, 
                 transforms='linear', tracking_transforms = 'linear', name = 'Imp', **kwargs):    

        super().__init__(params, X, y, session_path, simulation_setup, exp_format, metric, loss, threshold, minimize, yerr, weight, tracking_metric, tracking_loss, tracking_exp_format, tracking_X, tracking_y, tracking_weight, transforms, tracking_transforms, name, **kwargs)

        # ImpedanceAgent specific parameters
        self.f_min = f_min
        self.f_max = f_max
        self.f_steps = f_steps
        self.V_0 = V_0
        self.G_frac = G_frac
        self.del_V = del_V

    def validate_exp_format(self, exp_format):
        """Validate the exp_format parameter to ensure it is in the correct format for ImpedanceAgent

        Parameters
        ----------
        exp_format : str
            Format of the experimental data, must be one of the allowed formats for impedance data.

        Raises
        ------
        ValueError
            If the exp_format is not valid
        """            
        
        is_valid = (exp_format in ['Cf','Gf','CGf','Nyquist','BodeImZ','BodeReZ','Bode']) 
        if not is_valid:
            raise ValueError(f'{exp_format} is an invalid impedance format. Possible values are: Cf, Gf, CGf, Nyquist, BodeImZ, BodeReZ, Bode')
                    
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
        df = self.run_impedance_simulation(parameters)

        return self._run_Ax(df,self.reformat_impedance_data)
    
    def run_impedance_simulation(self, parameters):
        """Run the simulation with the parameters and return the simulated values

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.

        Returns
        -------
        dataframe
            Dataframe with the simulated impedance values.
        """    

        parallel = self.kwargs.get('parallel', False)
        max_jobs = self.kwargs.get('max_jobs', 1)
        output_file = self.kwargs.get('output_file', 'freqZ.dat')

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

        ret,mess = run_impedance_simu(self.simulation_setup, self.session_path, f_min = self.f_min, f_max = self.f_max, f_steps = self.f_steps, V_0 = self.V_0, G_frac = self.G_frac, del_V = self.del_V, UUID=UUID, cmd_pars=clean_pars, output_file = 'freqZ.dat',**dummy_kwargs)
        
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
            df = pd.read_csv(os.path.join(self.session_path, 'freqZ_'+UUID+'.dat'), sep=r'\s+')
        except:
            print('No impedance data found for UUID '+UUID + ' and cmd_pars '+str(cmd_pars))
            return np.nan

        return df

    def run(self, parameters,X=None,exp_format='Cf'):
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

        df = self.run_impedance_simulation(parameters)
        if df is np.nan or len(df) == 0:
            return np.nan

        if X is None:
            X = self.X[0]

        Xfit, yfit = self.reformat_impedance_data(df, X, exp_format)

        return yfit


    def reformat_impedance_data(self,df,X,exp_format='Cf'):
        """ Reformat the data depending on the exp_format and X values
        Also interpolates the data if the simulation did not return the same points as the experimental data (i.e. if some points did not converge)

        Parameters
        ----------
        df : dataframe
            Dataframe with the impedance dara from run_impedance_simulation function.
        X : array-like, optional
            1-D array containing the x axis values, by default None.
        exp_format : str, optional
            Format of the experimental data, by default 'Cf'.

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
        if exp_format == 'Cf':

            if len(X) == len(df['freq'].values):
                if np.allclose(X, np.asarray(df['freq'].values)):
                    do_interp = False
            
            if do_interp:
                # Do interpolation in case SIMsalabim did not return the same number of points 
                try:
                    if df['freq'].values[0] > df['freq'].values[-1]:
                        tck = interpolate.splrep(np.asarray(df['freq'].values)[::-1], np.asarray(df['C'].values)[::-1], s=0)   
                    else:
                        tck = interpolate.splrep(np.asarray(df['freq'].values), np.asarray(df['C'].values), s=0)
                    yfit = interpolate.splev(X, tck, der=0, ext=0)
                except:
                    f = interpolate.interp1d(df['freq'], df['C'], fill_value='extrapolate', kind='linear')
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    yfit = f(X)
            else:
                Xfit = X
                yfit = np.asarray(df['C'].values)

        elif exp_format == 'Gf':

            if len(X) == len(df['freq'].values):
                if np.allclose(X, np.asarray(df['freq'].values)):
                    do_interp = False
            
            if do_interp:
                # Do interpolation in case SIMsalabim did not return the same number of points 
                try:
                    if df['freq'].values[0] > df['freq'].values[-1]:
                        tck = interpolate.splrep(np.asarray(df['freq'].values)[::-1], np.asarray(df['G'].values)[::-1], s=0)
                    else:
                        tck = interpolate.splrep(np.asarray(df['freq'].values), np.asarray(df['G'].values), s=0)
                    yfit = interpolate.splev(X, tck, der=0, ext=0)
                except:
                    f = interpolate.interp1d(df['freq'], df['G'], fill_value='extrapolate', kind='linear')
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    yfit = f(X)
            else:
                Xfit = X
                yfit = np.asarray(df['G'].values)

        elif exp_format == 'CGf':
                freqs = self.kwargs.get('freqs',None)
                if freqs is None:
                    raise ValueError('freqs must be specified for Nyquist plot in case not all frequencies are returned by SIMsalabim')
                
                if len(X) == len(df['freq'].values):
                    if np.allclose(X, np.asarray(df['freq'].values)):
                        do_interp = False
                freqs_fit = np.asarray(df['freq'].values)
                if do_interp:
                    # Do interpolation in case SIMsalabim did not return the same number of points 
                    try:
                        if df['freq'].values[0] > df['freq'].values[-1]:
                            tck = interpolate.splrep(np.asarray(df['freq'].values)[::-1], np.asarray(df['C'].values)[::-1], s=0)
                            tck2 = interpolate.splrep(np.asarray(df['freq'].values)[::-1], np.asarray(df['G'].values)[::-1], s=0)
                        else:
                            tck = interpolate.splrep(np.asarray(df['freq'].values), np.asarray(df['C'].values), s=0)
                            tck2 = interpolate.splrep(np.asarray(df['freq'].values), np.asarray(df['G'].values), s=0)
                        yfit = interpolate.splev(freqs, tck, der=0, ext=0)
                        yfit2 = interpolate.splev(freqs, tck2, der=0, ext=0)
                        Xfit = np.concatenate((freqs,freqs))
                        yfit = np.concatenate((yfit,yfit2))
                    except:
                        f = interpolate.interp1d(df['freq'], df['C'], fill_value='extrapolate', kind='linear')
                        yfit = f(freqs)
                        f = interpolate.interp1d(df['freq'], df['G'], fill_value='extrapolate', kind='linear')
                        yfit2 = f(freqs)
                        warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                        # put C and G in the same array and double the length of Xfit
                        Xfit = np.concatenate((freqs,freqs))
                        yfit = np.concatenate((yfit,yfit2))
                else:
                    Xfit = np.concatenate((freqs,freqs))
                    yfit = np.concatenate((np.asarray(df['C'].values),np.asarray(df['G'].values)))

        elif exp_format == 'Bode':
            
            Xfit = np.asarray(df['freq'].values)
            yfit = np.asarray(df['ReZ'].values) 
            yfit2 = np.asarray(df['ImZ'].values)
            # yfit = np.concatenate((yfit,yfit2))
            # Xfit = np.concatenate((Xfit,Xfit))
            freqs_fit = np.asarray(df['freq'].values)

            if len(X) == len(Xfit):
                if np.allclose(X, Xfit):
                    do_interp = False
            
            if do_interp:
                freqs = self.kwargs.get('freqs',None)
                if freqs is None:
                    raise ValueError('freqs must be specified for Nyquist plot in case not all frequencies are returned by SIMsalabim')
                
                # Do interpolation in case SIMsalabim did not return the same number of points 
                try:
                    if freqs_fit[0] > freqs_fit[-1]:
                        freqs_fit = freqs_fit[::-1]
                        dum_Xfit = Xfit[::-1]
                        dum_yfit = yfit[::-1]
                        dum_yfit2 = yfit2[::-1]
                    else:
                        dum_Xfit = Xfit
                        dum_yfit = yfit
                        dum_yfit2 = yfit2

                    # interpolate ReZ
                    tck = interpolate.splrep(freqs_fit, dum_yfit, s=0)
                    yfit = interpolate.splev(freqs, tck, der=0, ext=0)
                    # interpolate ImZ
                    tck = interpolate.splrep(freqs_fit, dum_yfit2, s=0)
                    yfit2 = interpolate.splev(freqs, tck, der=0, ext=0)
                except:
                    f = interpolate.interp1d(freqs_fit, yfit, fill_value='extrapolate', kind='linear')
                    yfit = f(freqs)
                    f = interpolate.interp1d(freqs_fit, yfit2, fill_value='extrapolate', kind='linear')
                    yfit2 = f(freqs) 
                    # f = interpolate.interp1d(Xfit, yfit, fill_value='extrapolate', kind='linear')
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    # yfit = f(X)
            
            yfit = np.concatenate((yfit,yfit2))
            Xfit = np.concatenate((Xfit,Xfit))

        elif exp_format == 'BodeImZ':
            
            Xfit = np.asarray(df['freq'].values) 
            yfit = np.asarray(df['ImZ'].values)

            if len(X) == len(Xfit):
                if np.allclose(X, Xfit):
                    do_interp = False
            
            if do_interp:
                # Do interpolation in case SIMsalabim did not return the same number of points 
                try:
                    if Xfit[0] > Xfit[-1]: # check if the frequencies are in descending order and reverse them if necessary
                        dum_Xfit = Xfit[::-1]
                        dum_yfit = yfit[::-1]
                    else:
                        dum_Xfit = Xfit
                        dum_yfit = yfit
                    tck = interpolate.splrep(dum_Xfit, dum_yfit, s=0)
                except:
                    f = interpolate.interp1d(Xfit, yfit, fill_value='extrapolate', kind='linear')
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    yfit = f(X)

        elif exp_format == 'BodeReZ':

            Xfit = np.asarray(df['freq'].values) 
            yfit = np.asarray(df['ReZ'].values)

            if len(X) == len(Xfit):
                if np.allclose(X, Xfit):
                    do_interp = False
            
            if do_interp:
                # Do interpolation in case SIMsalabim did not return the same number of points 
                try:
                    if Xfit[0] > Xfit[-1]: # check if the frequencies are in descending order and reverse them if necessary
                        dum_Xfit = Xfit[::-1]
                        dum_yfit = yfit[::-1]
                    else:
                        dum_Xfit = Xfit
                        dum_yfit = yfit
                    tck = interpolate.splrep(dum_Xfit, dum_yfit, s=0)
                    yfit = interpolate.splev(X, tck, der=0, ext=0)
                except:
                    f = interpolate.interp1d(Xfit, yfit, fill_value='extrapolate', kind='linear')
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    yfit = f(X)

        elif exp_format == 'Nyquist':
            
            if self.metric.lower() == 'mmeud' or self.metric.lower() == 'dmeud':
                Xfit = np.asarray(df['ReZ'].values)
                yfit = np.asarray(df['ImZ'].values)
            else:
                raise ValueError('Invalid metric for Nyquist plot. Possible values are: mmeud, dmeud. This is because the other metrics results in the same optimization as Bode plot so it is not necessary to use Nyquist plot for those metrics')
                           
            
            if len(X) == len(Xfit):
                if np.allclose(X, Xfit):
                    do_interp = False

            if do_interp:
                freqs = self.kwargs.get('freqs',None)
                if freqs is None:
                    raise ValueError('freqs must be specified for Bode plot in case not all frequencies are returned by SIMsalabim')
                # Do interpolation in case SIMsalabim did not return the same number of points 
                try:
                    # interpolate ReZ
                    dum_freqs = np.asarray(df['freq'].values)
                    dum_Re = np.asarray(df['ReZ'].values)
                    dum_Im = np.asarray(df['ImZ'].values)
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
                    f = interpolate.interp1d(np.asarray(df['freq'].values), np.asarray(df['ReZ'].values), fill_value='extrapolate', kind='linear')
                    yfit = f(freqs)
                    f = interpolate.interp1d(np.asarray(df['freq'].values), np.asarray(df['ImZ'].values), fill_value='extrapolate', kind='linear')
                    yfit2 = f(freqs)
                    # put ReZ and ImZ in the same array and double the length of Xfit
                    Xfit = yfit
                    yfit = yfit2

                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
        else:
            raise ValueError('Invalid impedance format. Possible values are: Cf, Gf, Nyquist, BodeImZ, BodeReZ, Bode')

        return Xfit, yfit