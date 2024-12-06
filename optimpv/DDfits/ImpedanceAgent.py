"""ImpedanceAgent class for impedance simulations"""
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
        Fractional light intensity, by default 1.
    del_V : float, optional
        Voltage step for the impedance simulation, by default 0.01.
    simulation_setup : str, optional
        Path to the simulation setup file, if None then use the default file 'simulation_setup.txt'in the session_path directory, by default None.
    exp_format : str, optional
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
    name : str, optional
        Name of the agent, by default 'Imp'.
    **kwargs : dict
        Additional keyword arguments.
    """   
    def __init__(self, params, X, y, session_path, f_min, f_max, f_steps, V_0, G_frac = 1, del_V = 0.01, simulation_setup = None, exp_format = ['Cf'], metric = ['mse'], loss = ['linear'], threshold = [100], minimize = [True], yerr = None, weight = None, name = 'Imp', **kwargs):    

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
        self.f_min = f_min
        self.f_max = f_max
        self.f_steps = f_steps
        self.V_0 = V_0
        self.G_frac = G_frac
        self.del_V = del_V
        self.name = name
        
        self.exp_format = exp_format
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]

        # check that all elements in exp_format are valid
        for imp_form in self.exp_format:
            if imp_form not in ['Cf','Gf','CGf','Nyquist','BodeImZ','BodeReZ','Bode']:
                raise ValueError('{imp_form} is an invalid impedance format. Possible values are: Cf, Gf, CGf,Nyquist, BodeImZ, BodeReZ, Bode')

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
            Dictionary with the target metric value.
        """  
        df = self.run_impedance_simulation(parameters)
        if df is np.nan:
            dum_dict = {}
            for i in range(len(self.exp_format)):
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = np.nan
            return dum_dict
        
        dum_dict = {}

        for i in range(len(self.exp_format)):

            Xfit, yfit = self.reformat_impedance_data(df,self.X[i],exp_format=self.exp_format[i])
            
            dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(self.target_metric(self.y[i],yfit,self.metric[i],self.X[i],Xfit,weight=self.weight[i]),loss=self.loss[i])

        return dum_dict
    
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
        if df is np.nan:
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