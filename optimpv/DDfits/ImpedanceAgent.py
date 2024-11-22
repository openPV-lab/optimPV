"""JVAgent class for steady-state JV simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy
from scipy import interpolate

from optimpv import *
from optimpv.general.general import *
from optimpv.DDfits.SIMsalabimAgent import SIMsalabimAgent
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
        Fraction of the voltage to use for the G measurement, by default 1.
    simulation_setup : str, optional
        Path to the simulation setup file, if None then use the default file 'simulation_setup.txt'in the session_path directory, by default None.
    impedance_format : str, optional
        Format of the impedance data, possible values are: 'Cf', 'Gf', 'Nyquist', 'BodeImZ', 'BodeReZ', 'Bode', by default 'Cf'.
    metric : str, optional
        Metric to evaluate the model, see optimpv.general.calc_metric for options, by default 'mse'.
    loss : str, optional
        Loss function to use, see optimpv.general.loss_function for options, by default 'linear'.
    yerr : array-like, optional
        Errors in the current values, by default None.
    weight : array-like, optional
        Weights used for fitting if weight is None and yerr is not None, then weight = 1/yerr**2, by default None.
    **kwargs : dict
        Additional keyword arguments.
    """   
    def __init__(self, params, X, y, session_path, f_min, f_max, f_steps, V_0, G_frac=1, del_V=0.01, simulation_setup=None, impedance_format = 'Cf', metric = 'mse', loss = 'linear', yerr=None,weight=None,**kwargs):    

        self.params = params
        self.session_path = session_path  
        if simulation_setup is None:
            self.simulation_setup = os.path.join(session_path,'simulation_setup.txt')
        else:
            self.simulation_setup = simulation_setup  
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.yerr = np.asarray(yerr)
        self.metric = metric
        self.loss = loss
        self.kwargs = kwargs
        self.f_min = f_min
        self.f_max = f_max
        self.f_steps = f_steps
        self.V_0 = V_0
        self.G_frac = G_frac
        self.del_V = del_V
        
        if impedance_format not in ['Cf','Gf','Nyquist','BodeImZ','BodeReZ','Bode']:
            raise ValueError('Invalid impedance format. Possible values are: Cf, Gf, Nyquist, BodeImZ, BodeReZ, Bode')
        
        self.impedance_format = impedance_format

        if self.loss is None:
            self.loss = 'linear'

        if weight is not None:
            self.weight = np.asarray(weight)
        else:
            if yerr is not None:
                self.weight = 1/yerr**2
            else:
                self.weight = None

        while True: # need this to be thread safe
            try:
                dev_par, layers = load_device_parameters(session_path, simulation_setup, run_mode = False)
                break
            except:
                pass 
            time.sleep(0.002)
        
        self.dev_par = dev_par
        self.layers = layers
        SIMsalabim_params  = {}

        for layer in layers:
            SIMsalabim_params[layer[1]] = ReadParameterFile(os.path.join(session_path,layer[2]))

        self.SIMsalabim_params = SIMsalabim_params
        pnames = list(SIMsalabim_params[list(SIMsalabim_params.keys())[0]].keys())
        pnames = pnames + list(SIMsalabim_params[list(SIMsalabim_params.keys())[1]].keys())
        self.pnames = pnames    


    def target_metric(self,y,yfit,Xfit=None):
        """Calculate the target metric depending on self.metric

        Parameters
        ----------
        y : array-like
            1-D array containing the target values.
        yfit : array-like
            1-D array containing the fitted values.

        Returns
        -------
        float
            Target metric value.
        """        
        if self.metric.lower() == 'mmeud':
            if Xfit is None:
                raise ValueError('Xfit must be specified for the mmed metric')
            return mean_min_euclidean_distance(self.X,y,Xfit,yfit)
        elif self.metric.lower() == 'dmeud':
            if Xfit is None:
                raise ValueError('Xfit must be specified for the med metric')
            return direct_mean_euclidean_distance(self.X,y,Xfit,yfit)
        else:
            return  calc_metric(y,yfit,sample_weight=self.weight,metric_name=self.metric)
    

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

        if self.metric.lower() == 'mmeud' or self.metric.lower() == 'dmeud':
            Xfit,yfit = self.run(parameters)
            return {self.metric: loss_function(self.target_metric(self.y,yfit,Xfit),loss=self.loss)}
        else:
            yfit = self.run(parameters)
            y = self.y
            return {self.metric: loss_function(self.target_metric(y,yfit),loss=self.loss)}
    
    
    def run(self, parameters):
        """Run the simulation with the parameters and return the fitted current values

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.

        Returns
        -------
        array-like
            1-D array containing the simulated current values.
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

        # get Gfracs from X
        if len(self.X.shape) == 1:
            Gfracs = None
        else:
            Gfracs = np.unique(self.X[:,1])

        # prepare the cmd_pars for the simulation
        custom_pars, clean_pars, VarNames = self.prepare_cmd_pars(parameters, custom_pars, clean_pars, VarNames)

        # check if there are any custom_pars that are energy level offsets
        clean_pars = self.energy_level_offsets(custom_pars, clean_pars)

        # check if there are any duplicated parameters in cmd_pars
        self.check_duplicated_parameters(clean_pars)
        
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
            if not ret == 0 :
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

        # save data for fitting freq ReZ ImZ ReErrZ ImErrZ C G errC errG
        Xfit,yfit = [],[]
        do_interp = True
        try:
            df = pd.read_csv(os.path.join(self.session_path, 'freqZ_'+UUID+'.dat'), sep=r'\s+')
        except:
            print('No impedance data found for UUID '+UUID + ' and cmd_pars '+str(cmd_pars))
            return np.nan

        if self.impedance_format == 'Cf':

            if len(self.X) == len(df['freq'].values):
                if np.allclose(self.X, np.asarray(df['freq'].values)):
                    do_interp = False
            
            if do_interp:
                # Do interpolation in case SIMsalabim did not return the same number of points 
                try:
                    if df['freq'].values[0] > df['freq'].values[-1]:
                        tck = interpolate.splrep(np.asarray(df['freq'].values)[::-1], np.asarray(df['C'].values)[::-1], s=0)   
                    else:
                        tck = interpolate.splrep(np.asarray(df['freq'].values), np.asarray(df['C'].values), s=0)
                    yfit = interpolate.splev(self.X, tck, der=0, ext=0)
                except:
                    f = interpolate.interp1d(df['freq'], df['C'], fill_value='extrapolate', kind='linear')
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    yfit = f(self.X)
            else:
                Xfit = self.X
                yfit = np.asarray(df['C'].values)
        elif self.impedance_format == 'Gf':

            if len(self.X) == len(df['freq'].values):
                if np.allclose(self.X, np.asarray(df['freq'].values)):
                    do_interp = False
            
            if do_interp:
                # Do interpolation in case SIMsalabim did not return the same number of points 
                try:
                    if df['freq'].values[0] > df['freq'].values[-1]:
                        tck = interpolate.splrep(np.asarray(df['freq'].values)[::-1], np.asarray(df['G'].values)[::-1], s=0)
                    else:
                        tck = interpolate.splrep(np.asarray(df['freq'].values), np.asarray(df['G'].values), s=0)
                    yfit = interpolate.splev(self.X, tck, der=0, ext=0)
                except:
                    f = interpolate.interp1d(df['freq'], df['G'], fill_value='extrapolate', kind='linear')
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    yfit = f(self.X)
            else:
                Xfit = self.X
                yfit = np.asarray(df['G'].values)
        elif self.impedance_format == 'Nyquist':

            Xfit = np.asarray(df['ReZ'].values) 
            yfit = np.asarray(df['ImZ'].values)
            freqs_fit = np.asarray(df['freq'].values)
            if len(self.X) == len(Xfit):
                if np.allclose(self.X, Xfit):
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
                    else:
                        dum_Xfit = Xfit
                        dum_yfit = yfit
                    # interpolate ReZ
                    tck = interpolate.splrep(freqs_fit, dum_Xfit, s=0)
                    Xfit = interpolate.splev(freqs, tck, der=0, ext=0)
                    # interpolate ImZ
                    tck = interpolate.splrep(freqs_fit, dum_yfit, s=0)
                    yfit = interpolate.splev(freqs, tck, der=0, ext=0)
                except:
                    f = interpolate.interp1d(freqs_fit, Xfit, fill_value='extrapolate', kind='linear')
                    Xfit = f(freqs)
                    f = interpolate.interp1d(freqs_fit, yfit, fill_value='extrapolate', kind='linear')
                    yfit = f(freqs) 
                    # f = interpolate.interp1d(Xfit, yfit, fill_value='extrapolate', kind='linear')
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    # yfit = f(self.X)

        elif self.impedance_format == 'BodeImZ':
            
            Xfit = np.asarray(df['freq'].values) 
            yfit = np.asarray(df['ImZ'].values)

            if len(self.X) == len(Xfit):
                if np.allclose(self.X, Xfit):
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
                    yfit = f(self.X)
        elif self.impedance_format == 'BodeReZ':

            Xfit = np.asarray(df['freq'].values) 
            yfit = np.asarray(df['ReZ'].values)

            if len(self.X) == len(Xfit):
                if np.allclose(self.X, Xfit):
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
                    yfit = interpolate.splev(self.X, tck, der=0, ext=0)
                except:
                    f = interpolate.interp1d(Xfit, yfit, fill_value='extrapolate', kind='linear')
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    yfit = f(self.X)
        elif self.impedance_format == 'Bode':
            
            Xfit = np.asarray(df['freq'].values) 
            yfit = np.asarray(df['ReZ'].values)
            yfit2 = np.asarray(df['ImZ'].values)
            # put ReZ and ImZ in the same array and double the length of Xfit
            Xfit = np.repeat(Xfit,2)
            yfit = np.abs(np.concatenate((yfit,yfit2)))

            if self.metric.lower() == 'mmeud' or self.metric.lower() == 'dmeud':
                return Xfit, yfit
            
            if len(self.X) == len(Xfit):
                if np.allclose(self.X, Xfit):
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

                    # put ReZ and ImZ in the same array and double the length of Xfit
                    Xfit = np.repeat(freqs,2)
                    yfit = np.abs(np.concatenate((yfit,yfit2)))
                    
                except Exception as e:
                    f = interpolate.interp1d(np.asarray(df['freq'].values), np.asarray(df['ReZ'].values), fill_value='extrapolate', kind='linear')
                    yfit = f(freqs)
                    f = interpolate.interp1d(np.asarray(df['freq'].values), np.asarray(df['ImZ'].values), fill_value='extrapolate', kind='linear')
                    yfit2 = f(freqs)
                    # put ReZ and ImZ in the same array and double the length of Xfit
                    Xfit = np.repeat(freqs,2)
                    yfit = np.abs(np.concatenate((yfit,yfit2)))
                    print(e)
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
            
        else:
            raise ValueError('Invalid impedance format. Possible values are: Cf, Gf, Nyquist, BodeImZ, BodeReZ, Bode')

        return yfit