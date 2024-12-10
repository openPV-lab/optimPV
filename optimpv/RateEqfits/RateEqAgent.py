"""Provides general functionality for Agent objects for non ideal diode simulations"""
######### Package Imports #########################################################################

import os, uuid, sys, copy, warnings
import numpy as np
import pandas as pd
from scipy import interpolate, constants

try: 
    import pvlib
    from pvlib.pvsystem import i_from_v
    got_pvlib = True
except:
    got_pvlib = False
    warnings.warn('pvlib not installed, using scipy for diode equation')

from optimpv import *
from optimpv.general.general import calc_metric, loss_function
from optimpv.general.BaseAgent import BaseAgent
from optimpv.RateEqfits.RateEqModel import *
from optimpv.RateEqfits.Pumps import *

## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')

######### Agent Definition #######################################################################
class RateEqAgent(BaseAgent):
    """Agent object for fitting rate equation models. 
    Available models are:  

    - Bimolecular-Trapping (BT)
    - Bimolecular-Trapping-detrapping (BTD)

    see optimpv.RateEqfits.RateEqModel.py for more details about the available models

    Parameters
    ----------
    params : list of Fitparam() objects
        List of Fitparam() objects.
    X : array-like
        1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values.
    y : array-like
        1-D array containing the current values.
    pump_model : function, optional
        Function to get the generated carrier density profile, by default initial_carrier_density.
    pump_args : dict, optional
        Arguments for the pump_model function, by default {}.
    exp_format : str or list of str, optional
        Format of the experimental data, cane be ['trPL','TAS','trMC'], by default 'trPL'.
    exp_format_agrs : dict, optional
        Arguments dependent on the exp_format function, by default {}.
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
        Name of the agent, by default 'RateEq'.
    **kwargs : dict
        Additional keyword arguments.
    """    
    def __init__(self, params, X, y, model = BT_model, pump_model = initial_carrier_density, pump_args = {}, exp_format = 'trPL', fixed_model_args = {}, metric = 'mse', loss = 'linear', threshold = 100, minimize = True, yerr = None, weight = None, name = 'RateEq', equilibrate = True, detection_limit=None,**kwargs):

        self.params = params
        self.X = X # voltage and Gfrac
        self.y = y
        self.pump_model = pump_model
        self.pump_args = pump_args
        self.model = model
        self.fixed_model_args = fixed_model_args
        self.metric = metric
        self.loss = loss
        self.threshold = threshold
        self.minimize = minimize
        self.equilibrate = equilibrate
        self.detection_limit = detection_limit

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
        
        self.yerr = yerr
        self.weight = weight
        self.name = name
        self.kwargs = kwargs

        self.exp_format = exp_format
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]

        # check that all elements in exp_format are valid
        for form in self.exp_format:
            if form not in ['trPL','TAS','trMC']:
                raise ValueError(f'{form} is an invalid exp_format, must be either "trPL", "TAS" or "trMC"')

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

        self.compare_logs = self.kwargs.get('compare_logs',False) 
        if 'compare_logs' in self.kwargs.keys():
            # remove it from kwargs
            self.kwargs.pop('compare_logs')
    
    def run_RateEq(self,parameters):
        """Run the diode model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        float
            Loss function value.
        """    

        # get Gfracs from X
        if 'QE' in parameters.keys():
            QE = parameters['QE']
        elif 'QE' in self.fixed_model_args.keys():
            QE = self.fixed_model_args['QE']
        else:
            QE = 1

        Gfracs = []
        got_gfrac_none = False
        len_X = len(self.X[0])
        x_default = self.X[0]
        for xx in self.X:
            if len(xx.shape) == 1:
                Gfracs = None
                got_gfrac_none = True
                if len(xx) != len_X:
                    raise ValueError('all X elements should have the same shape')
                if not np.allclose(xx,x_default):
                    raise ValueError('all X elements should be the same, if they are different then create a separate agent for each X element')
            else:
                if got_gfrac_none:
                    raise ValueError('all X elements should have the same shape and Gfrac should be provided for all elements if specified for one')
                # append np.unique(xx[:,1]) to Gfracs list
                if len(xx.shape) == 2:
                    Gfrac = np.unique(xx[:,1])
                    for g in Gfrac:
                        if g not in Gfracs:
                            Gfracs.append(g)
                else:
                    Gfracs = None
                if len(xx) != len_X:
                    raise ValueError('all X elements should have the same shape')

                # check all elements in xx[:,0] are the same
                for Gfrac in Gfracs:
                    if not np.allclose(xx[xx[:,1]==Gfrac,0],x_default[x_default[:,1]==Gfrac,0]):
                # if not np.allclose(xx[:,0],x_default[:,0]):
                        raise ValueError('all X elements should be the same, if they are different then create a separate agent for each X element')

        if Gfracs is not None:
            Gfracs = np.asarray(Gfracs)    
        ns, ps = None, None
        if Gfracs is None:
            t = self.X[0] # time axis should be the same for all elements
            tmax = 0.99999*1/self.pump_args['fpu'] # maximum time for the pump
            t_span = np.linspace(0,tmax,len(t)) # time axis for the simulation, here we need a different time axis for the simulation in case there is any equilibration to make sure that the full pulse is included and to reproduce the accumulated carrier density properly

            # get the pump profile
            Generation = self.pump_model(t, **self.pump_args) * QE # multiply by the quantum efficiency

            if 'N0' in self.pump_args.keys():
                N0 = self.pump_args['N0']
            else:
                N0 = 0

            if 'G_frac' in self.pump_args.keys():
                G_frac = self.pump_args['G_frac']
            else:
                G_frac = 1

            ns, ps = self.model(parameters, t, Generation, t_span, N0 = N0, equilibrate = self.equilibrate, G_frac = G_frac, **self.kwargs)
            Gfrac_list = np.ones(len(t))
        else:
            for Gfrac in Gfracs:
                t = self.X[0][self.X[0][:,1] == Gfrac,0]
                tmax = 0.99999*1/self.pump_args['fpu']
                t_span = t
                
                if t_span[-1] < tmax:
                    dum = np.linspace(t[-1],tmax,100)
                    dum = dum[1:]
                    t_span = np.hstack((t,dum))

                Generation = self.pump_model(t_span, G_frac = Gfrac, **self.pump_args) * QE
                
                if 'N0' in self.pump_args.keys():
                    N0 = self.pump_args['N0']
                else:
                    N0 = 0

                ns_, ps_ = self.model(parameters, t, Generation, t_span, N0 = N0, equilibrate = self.equilibrate, G_frac = Gfrac, **self.kwargs)
                
                if ns is None:
                    ns = ns_
                    ps = ps_
                    t_list = t
                    Gfrac_list = np.ones(len(t))*Gfrac
                else:
                    ns = np.hstack((ns,ns_))
                    ps = np.hstack((ps,ps_))
                    t_list = np.hstack((t_list,t))
                    Gfrac_list = np.hstack((Gfrac_list,np.ones(len(t))*Gfrac))

        dum_dict = {}
        dum_dict['n'] = ns
        dum_dict['p'] = ps
        if Gfracs is None:
            dum_dict['t'] = t
            dum_dict['G_frac'] = Gfrac_list
        else:
            dum_dict['t'] = t_list
            dum_dict['G_frac'] = Gfrac_list
        try:
            df = pd.DataFrame(dum_dict)
        except:
            for key in dum_dict.keys():
                print(key,len(dum_dict[key]))

        return df
    
    def reformat_data(self,df,X,parameters,exp_format='trPL'):
        """Reformat the data to the experimental format

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the simulation results.
        X : array-like
            1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values.
        parameters : dict
            Dictionary of parameter names and values.
        exp_format : str, optional
            Format of the experimental data, cane be ['trPL','TAS','trMC'], by default 'trPL'.

        Returns
        -------
        DataFrame
            DataFrame containing the simulation results in the experimental format.
        """  

        # check if Gfrac is None is unique in df
        Gfracs = np.unique(df['G_frac'])

        if len(Gfracs) == 1:
            Gfracs = None 

        Xfit,yfit = None,None # will be np.array
        do_interp = True

        if exp_format == 'trPL': # transient photoluminescence
            if 'k_direct' in parameters.keys():
                k_direct = parameters['k_direct']
            elif 'k_direct' in self.fixed_model_args.keys():
                k_direct = self.fixed_model_args['k_direct']
            else:
                raise ValueError('k_direct should be provided in the parameters or fixed_model_args')
            
            if 'I_factor_PL' in parameters.keys():
                I_factor = parameters['I_factor_PL']   
            elif 'I_factor_PL' in self.fixed_model_args.keys():
                I_factor = self.fixed_model_args['I_factor_PL']
            else:
                raise ValueError('I_factor_PL should be provided in the parameters or fixed_model_args')
            
            if 'N_A' in parameters.keys():
                N_A = parameters['N_A']
            elif 'N_A' in self.fixed_model_args.keys():
                N_A = self.fixed_model_args['N_A']
            else:
                N_A = 0

            if Gfracs is None:
                signal = I_factor * k_direct * df['n'] * (df['p'] + N_A) 
                t = df['t']

                # check if t = X
                if len(t) == len(X):
                    if np.allclose(t,X):
                        do_interp = False

                if do_interp:
                    try:
                        tck = interpolate.splrep(t,signal)
                        signal = interpolate.splev(X,tck)
                    except:
                        f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                        yfit = f(X)
                else:
                    yfit = np.asarray(signal)
                Xfit = np.asarray(X)
            else:
                for Gfrac in Gfracs:
                    dum_df = df[df['G_frac'] == Gfrac]
                    signal = I_factor * k_direct * dum_df['n'] * (dum_df['p'] + N_A) 
                    t = dum_df['t']
                    X_ = X[X[:,1] == Gfrac,0]

                    if do_interp:
                        try:
                            tck = interpolate.splrep(t,signal)
                            yfit_ = interpolate.splev(X_,tck)
                        except Exception as e:
                            f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                            yfit_ = f(X_)
                    else:
                        yfit_ = signal

                    if yfit is None:
                        yfit = np.asarray(yfit_)
                    else:
                        yfit = np.hstack((yfit,yfit_))

                    if Xfit is None:
                        Xfit = X
                    else:
                        Xfit = np.hstack((Xfit,X))

                Xfit = np.asarray(Xfit)
                yfit = np.asarray(yfit)
        elif exp_format == 'trMC': # transient microwave conductivity

            if 'I_factor_MC' in parameters.keys():
                I_factor = parameters['I_factor_MC']
            elif 'I_factor_MC' in self.fixed_model_args.keys():
                I_factor = self.fixed_model_args['I_factor_MC']
            else:
                raise ValueError('I_factor_MC should be provided in the parameters or fixed_model_args')
                
            if 'ratio_mu' in parameters.keys():
                ratio_mu = parameters['ratio_mu']
            elif 'ratio_mu' in self.fixed_model_args.keys():
                ratio_mu = self.fixed_model_args['ratio_mu']
            else:
                raise ValueError('ratio_mu should be provided in the parameters or fixed_model_args')

            if Gfracs is None:
                signal = I_factor*(ratio_mu*df['n'] + df['p'])
                t = df['t']

                # check if t = X
                if len(t) == len(X):
                    if np.allclose(t,X):
                        do_interp = False

                if do_interp:
                    try:
                        tck = interpolate.splrep(t,signal)
                        signal = interpolate.splev(X,tck)
                    except:
                        f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                        yfit = f(X)
                else:
                    yfit = np.asarray(signal)
                Xfit = np.asarray(X)
            else:
                for Gfrac in Gfracs:
                    dum_df = df[df['G_frac'] == Gfrac]
                    signal = I_factor*(ratio_mu*dum_df['n'] + dum_df['p'])
                    t = dum_df['t']
                    X_ = X[X[:,1] == Gfrac,0]

                    if do_interp:
                        try:
                            tck = interpolate.splrep(t,signal)
                            yfit_ = interpolate.splev(X_,tck)
                        except Exception as e:
                            f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                            yfit_ = f(X_)
                    else:
                        yfit_ = signal

                    if yfit is None:
                        yfit = np.asarray(yfit_)
                    else:
                        yfit = np.hstack((yfit,yfit_))

                    if Xfit is None:
                        Xfit = X
                    else:
                        Xfit = np.hstack((Xfit,X))

                Xfit = np.asarray(Xfit)
                yfit = np.asarray(yfit)
        elif exp_format == 'TAS': # transient absorption spectroscopy
                
                if 'I_factor_TAS' in parameters.keys():
                    I_factor = parameters['I_factor_TAS']
                elif 'I_factor_TAS' in self.fixed_model_args.keys():
                    I_factor = self.fixed_model_args['I_factor_TAS']
                else:
                    I_factor = 1 # we usually don't have a factor for TAS and correct with cross-sections and thickness
                    # raise ValueError('I_factor_TAS should be provided in the parameters or fixed_model_args')
                    
                if 'cross_section' in parameters.keys():
                    cross_section = parameters['cross_section']
                elif 'cross_section' in self.fixed_model_args.keys():
                    cross_section = self.fixed_model_args['cross_section']
                else:
                    raise ValueError('cross_section should be provided in the parameters or fixed_model_args')
                
                if 'L' in parameters.keys():
                    L = parameters['L']
                elif 'L' in self.fixed_model_args.keys():
                    L = self.fixed_model_args['L']
                else:
                    raise ValueError('L should be provided in the parameters or fixed_model_args')
                                
                if Gfracs is None:
                    signal = df['n']
                    t = df['t']
    
                    # check if t = X
                    if len(t) == len(X):
                        if np.allclose(t,X):
                            do_interp = False
    
                    if do_interp:
                        try:
                            tck = interpolate.splrep(t,signal)
                            signal = interpolate.splev(X,tck)
                        except:
                            f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                            yfit = f(X)
                    else:
                        yfit = np.asarray(signal)

                    yfit -= np.mean(yfit) # simulate AC coupling 
                    yfit = I_factor*(np.exp(-cross_section*L*yfit) - 1) # \Delta T/T = -\sigma*L*\Delta \n - 1 (the minus one comes from the fact that we are looking at the change in transmission and since Beer-Lambert law is A = exp(-\sigma*L*n) 

                    Xfit = np.asarray(X)
                else:
                    for Gfrac in Gfracs:
                        dum_df = df[df['G_frac'] == Gfrac]
                        signal = dum_df['n']
                        t = dum_df['t']
                        X_ = X[X[:,1] == Gfrac,0]
    
                        if do_interp:
                            try:
                                tck = interpolate.splrep(t,signal)
                                yfit_ = interpolate.splev(X_,tck)
                            except Exception as e:
                                f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                                yfit_ = f(X_)
                        else:
                            yfit_ = signal

                        yfit_ -= np.mean(yfit_) # simulate AC coupling
                        yfit_ = I_factor*(np.exp(-cross_section*L*yfit_) - 1) # \Delta T/T = -\sigma*L*\Delta \n

                        if yfit is None:
                            yfit = np.asarray(yfit_)
                        else:
                            yfit = np.hstack((yfit,yfit_))
    
                        if Xfit is None:
                            Xfit = X
                        else:
                            Xfit = np.hstack((Xfit,X))
    
                    Xfit = np.asarray(Xfit)
                    
        if self.detection_limit is not None:
            yfit[yfit < self.detection_limit] = self.detection_limit

        return Xfit,yfit

    def run(self,parameters,X=None,exp_format='trPL'):
        """Run the diode model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        array-like
            Array containing the current values.
        """    

        parameters_rescaled = self.params_rescale(parameters, self.params)
        
        df = self.run_RateEq(parameters_rescaled)

        if df is np.nan:
            return np.nan

        if X is None:
            X = self.X[0]
        
        Xfit, yfit = self.reformat_data(df, X, parameters_rescaled, exp_format)
        
        return yfit


    def run_Ax(self,parameters):
        """Run the diode model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        float
            Loss function value.
        """    
        
        
        parameters_rescaled = self.params_rescale(parameters, self.params)
        
        df = self.run_RateEq(parameters_rescaled)


        if df is np.nan:
            dum_dict = {}
            for i in range(len(self.exp_format)):
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = np.nan
            return dum_dict
        
        dum_dict = {}
        
        for i in range(len(self.exp_format)):
            Xfit, yfit = self.reformat_data(df, self.X[i], parameters_rescaled, self.exp_format[i])
            if self.compare_logs:
                epsilon = np.finfo(np.float64).eps
                # if 0 in yfit, then add epsilon to avoid log(0)
                yfit[abs(yfit) <= epsilon] = epsilon
                y_ = copy.deepcopy(self.y[i])
                y_[abs(y_) <= epsilon] = epsilon
                
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(calc_metric(np.log10(abs(y_)),np.log10(abs(yfit)),sample_weight=self.weight[i],metric_name=self.metric[i]),loss=self.loss[i])
            else:
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(calc_metric(self.y[i],yfit,sample_weight=self.weight[i],metric_name=self.metric[i]),loss=self.loss[i])

        return dum_dict