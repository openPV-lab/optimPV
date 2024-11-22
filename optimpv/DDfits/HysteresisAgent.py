"""JVAgent class for steady-state JV simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy
from scipy import interpolate

from optimpv import *
from optimpv.general.general import calc_metric, loss_function
from optimpv.DDfits.SIMsalabimAgent import SIMsalabimAgent
from pySIMsalabim import *
from pySIMsalabim.experiments.impedance import *

######### Agent Definition #######################################################################
class HysteresisAgent(SIMsalabimAgent):  
    """HysteresisAgent class for impedance simulations with SIMsalabim

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
        Minimum frequency for the hysteresis simulation.
    f_max : float
        Maximum frequency for the hysteresis simulation.
    f_steps : int
        Number of frequency steps for the hysteresis simulation.
    V_0 : float
        Initial voltage for the hysteresis simulation.
    G_frac : float, optional
        Fraction of the voltage to use for the G measurement, by default 1.
    simulation_setup : str, optional
        Path to the simulation setup file, if None then use the default file 'simulation_setup.txt'in the session_path directory, by default None.
    hysteresis_format : str, optional
        Format of the hysteresis data, possible values are: 'Cf', 'Gf', 'Nyquist', 'BodeImZ', 'BodeReZ', 'Bode', by default 'Cf'.
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
    def __init__(self, params, X, y, session_path, Vmin, Vmax, scan_speed, steps, G_frac=1, direction = 1, simulation_setup=None, metric = 'mse', loss = 'linear', yerr=None,weight=None,**kwargs):    

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
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.scan_speed = scan_speed
        self.steps = steps
        self.G_frac = G_frac
        self.direction = direction
        
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


    def target_metric(self,y,yfit):
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

        
        # ret,mess = run_hysteresis_simu(self.simulation_setup, self.session_path, f_min = self.f_min, f_max = self.f_max, f_steps = self.f_steps, V_0 = self.V_0, G_frac = self.G_frac, del_V = self.del_V, UUID=UUID, cmd_pars=clean_pars, output_file = 'freqZ.dat',**dummy_kwargs)

        ret,mess = Hysteresis_JV(self.simulation_setup, self.session_path, False, self.scan_speed, direction, G_frac, tVG_name='tVG.txt', tj_name = 'tj.dat',varFile='none',run_mode=False, Vmin=0.0, Vmax=0.0, steps = self.steps, UUID=UUID, cmd_pars=clean_pars,**dummy_kwargs)
        
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
            df = pd.read_csv(os.path.join(self.session_path, 'tj_'+UUID+'.dat'), sep=r'\s+')
        except:
            print('No hysteresis data found for UUID '+UUID + ' and cmd_pars '+str(cmd_pars))
            return np.nan

        if len(self.X) == len(df['Vext'].values):
            if np.allclose(self.X, np.asarray(df['Vext'].values)):
                do_interp = False

        if do_interp:
            # calcuate time for each voltage step
            t_sim = df['t'].values
            Vext = df['Vext'].values
            # know the scan speed and voltage values so can calculate the time for each voltage step
            t_exp = np.zeros(len(Vext))
            t_exp[0] = 0
            for i in range(1,len(Vext)):
                t_exp[i] = t_exp[i-1] + abs(Vext[i]-Vext[i-1])/self.scan_speed
            
            # interpolate the data
            try:
                tck = interpolate.splrep(t_sim, df['Jext'].values, s=0)
                yfit = interpolate.splev(t_exp, tck, der=0)
            except:
                f = interpolate.interp1d(t_sim, df['Jext'].values, kind='linear', fill_value='extrapolate')
                yfit = f(t_exp)

        else:
            yfit = df['Jext'].values

        return np.asarray(yfit)