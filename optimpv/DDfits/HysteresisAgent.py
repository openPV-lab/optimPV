"""HysteresisAgent class for transient hysteresis JV simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy
from scipy import interpolate

from optimpv import *
from optimpv.general.general import calc_metric, loss_function
from optimpv.DDfits.SIMsalabimAgent import SIMsalabimAgent
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
    exp_format : str, optional
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
    name : str, optional
        Name of the agent, by default 'Hyst'.
    **kwargs : dict
        Additional keyword arguments.
    """   
    def __init__(self, params, X, y, session_path, Vmin=0, Vmax=1.2, scan_speed=0.1, steps=100, direction=1, G_frac=1, simulation_setup=None, exp_format='JV', metric='mse', loss='linear', threshold=100, minimize=True, yerr=None, weight=None, name='Hyst', **kwargs):    

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
        if df is np.nan:
            dum_dict = {}
            for i in range(len(self.exp_format)):
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = np.nan
            return dum_dict
        
        dum_dict = {}

        for i in range(len(self.exp_format)):

            Xfit, yfit = self.reformat_hysteresis_data(df,self.X[i],exp_format=self.exp_format[i])
            
            dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(self.target_metric(self.y[i],yfit,self.metric[i],self.X[i],Xfit,weight=self.weight[i]),loss=self.loss[i])

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

        ret, mess, rms = Hysteresis_JV(self.simulation_setup, self.session_path, 0, scan_speed=self.scan_speed, direction=self.direction, G_frac=self.G_frac, Vmin=self.Vmin, Vmax=self.Vmax, steps=self.steps, UUID=UUID, cmd_pars=clean_pars, tj_name= 'tj.dat', **dummy_kwargs)
        
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