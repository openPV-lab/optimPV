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
    def __init__(self, params, X, y, session_path, simulation_setup=None, metric = 'mse', loss = 'linear', yerr=None,weight=None,**kwargs):       
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
            1-D array containing the current values.
        yfit : array-like
            1-D array containing the fitted current values.

        Returns
        -------
        float
            Target metric value.
        """        

        if self.metric.lower() == 'intdiff':
            if len(self.X.shape) == 1:
                metric = np.trapz(np.abs(y-yfit),x=self.X[:,0])
            else:
                Gfracs = np.unique(self.X[:,1])
                metric = 0
                for Gfrac in Gfracs:
                    Jmin = min(np.min(y[self.X[:,1]==Gfrac]),np.min(yfit[self.X[:,1]==Gfrac]))
                    Jmax = max(np.max(y[self.X[:,1]==Gfrac]),np.max(yfit[self.X[:,1]==Gfrac]))
                    Vmin = min(np.min(self.X[self.X[:,1]==Gfrac,0]),np.min(self.X[self.X[:,1]==Gfrac,0]))
                    Vmax = max(np.max(self.X[self.X[:,1]==Gfrac,0]),np.max(self.X[self.X[:,1]==Gfrac,0]))
                    metric += np.trapz(np.abs(y[self.X[:,1]==Gfrac]-yfit[self.X[:,1]==Gfrac]),x=self.X[self.X[:,1]==Gfrac,0]) / ((Jmax-Jmin)*(Vmax-Vmin))
            return metric
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

        yfit = self.run(parameters)
        y = self.y
       
        return {self.metric: loss_function(self.target_metric(y,yfit),loss=self.loss)}
    
    def run_scikit(self, parameters):

        # create dictionary with parameters
        par_dict = {}
        idx = 0
        for param in self.params:
            if param.type == 'fixed':
                par_dict[param.name] = param.value
            else:
                par_dict[param.name] = parameters[idx]
                idx += 1
        print(par_dict)
        yfit = self.run(par_dict)
        y = self.y
        return loss_function(self.target_metric(y,yfit),loss=self.loss)
    
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
        
        ret, mess = run_SS_JV(self.simulation_setup, self.session_path, JV_file_name = 'JV.dat', G_fracs = Gfracs, UUID=UUID, cmd_pars=clean_pars, parallel = parallel, max_jobs = max_jobs)

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

        # save data for fitting
        Xfit,yfit = [],[]
        if Gfracs is None:
            try:
                df = pd.read_csv(os.path.join(self.session_path, 'JV_'+UUID+'.dat'), sep=r'\s+')
            except:
                print('No JV data found for UUID '+UUID + ' and cmd_pars '+str(cmd_pars))
                return np.nan

            do_interp = True
            if len(self.X) == len(df['Vext'].values):
                if np.allclose(self.X[:,0], np.asarray(df['Vext'].values)):
                    do_interp = False

            #check if all points from X[:,0] and data['Vext'].values are the same                
            if do_interp:
                # Do interpolation in case SIMsalabim did not return the same number of points 
                try:
                    tck = interpolate.splrep(np.asarray(df['Vext'].values), np.asarray(df['Jext'].values), s=0)
                    yfit = interpolate.splev(self.X, tck, der=0, ext=0)
                except: # if linear interpolation fails, use the minimum value of the JV curve
                    if min(self.X)- 0.025 < min(df['Vext']): # need this as a safety to make sure we output something
                        # add a point at the beginning of the JV curve
                        df = df.append({'Vext':min(self.X),'Jext':df['Jext'].iloc[0]},ignore_index=True)
                        df = df.sort_values(by=['Vext'])
                    f = interpolate.interp1d(df['Vext'], df['Jext'], fill_value='extrapolate', kind='linear')
                    yfit = f(self.X)
            else:
                Xfit = self.X
                yfit = np.asarray(df['Jext'].values)
        else:
            for Gfrac in Gfracs:
                try:
                    df = pd.read_csv(os.path.join(self.session_path, 'JV_Gfrac_'+str(Gfrac)+'_'+UUID+'.dat'), sep=r'\s+')
                except:
                    print('No JV data found for UUID '+UUID + ' and cmd_pars '+str(cmd_pars))
                    return np.nan
                
                Vext = np.asarray(df['Vext'].values)
                Jext = np.asarray(df['Jext'].values)
                G = np.ones_like(Vext)*Gfrac

                # check if all points from X[:,0] and data['Vext'].values are the same
                do_interp = True
                if len(self.X[self.X[:,1]==Gfrac,0]) == len(Vext) :
                    if np.allclose(self.X[self.X[:,1]==Gfrac,0], Vext):
                        do_interp = False

                if do_interp:
                    # Do interpolation in case SIMsalabim did not return the same number of points 
                    try:
                        tck = interpolate.splrep(Vext, Jext, s=0)
                        if len(Xfit) == 0:
                            Xfit = np.vstack((Vext,G)).T
                            yfit = interpolate.splev(self.X[self.X[:,1]==Gfrac,0], tck, der=0, ext=0)
                        else:
                            Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                            yfit = np.hstack((yfit,interpolate.splev(self.X[self.X[:,1]==Gfrac,0], tck, der=0, ext=0)))
                        
                    except:
                        if min(self.X[self.X[:,1]==Gfrac,0])- 0.025 < min(Vext):
                            # add a point at the beginning of the JV curve
                            df = df.append({'Vext':min(self.X[self.X[:,1]==Gfrac,0]),'Jext':df['Jext'].iloc[0]},ignore_index=True)
                            df = df.sort_values(by=['Vext'])
                        f = interpolate.interp1d(df['Vext'], df['Jext'], fill_value='extrapolate', kind='linear')
                        if len(Xfit) == 0:
                            Xfit = np.vstack((Vext,G)).T
                            yfit = f(self.X[self.X[:,1]==Gfrac,0])
                        else:
                            Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                            yfit = np.hstack((yfit,f(self.X[self.X[:,1]==Gfrac,0])))
                else:
                    if len(Xfit) == 0:
                        Xfit = np.vstack((Vext,G)).T
                        yfit = Jext
                    else:
                        Xfit = np.vstack((Xfit,np.vstack((Vext,G)).T))
                        yfit = np.hstack((yfit,Jext))
        
        return yfit