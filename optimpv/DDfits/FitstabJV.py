import numpy as np
import pandas as pd
import os, uuid, sys, copy
from scipy import interpolate

from optimpv import *
from optimpv.general.general import calc_metric, loss_function
from pySIMsalabim import *
from pySIMsalabim.experiments.JV_steady_state import *

# from ray import train, tune

class JV_SS_agent():
    def __init__(self, params, X, y, session_path, simss_device_parameters=None, metric = 'mse', loss = 'linear', yerr=None,weight=None,**kwargs):
        self.params = params
        self.session_path = session_path  
        if simss_device_parameters is None:
            self.simss_device_parameters = os.path.join(session_path,'simulation_setup.txt')
        else:
            self.simss_device_parameters = simss_device_parameters  
        self.X = X
        self.y = y
        self.yerr = yerr
        self.metric = metric
        self.loss = loss
        self.kwargs = kwargs

        if self.loss is None:
            self.loss = 'linear'

        if weight is not None:
            self.weight = weight
        else:
            if yerr is not None:
                self.weight = 1/yerr**2
            else:
                self.weight = None

        while True: # need this to be thread safe
            try:
                dev_par, layers = load_device_parameters(session_path, simss_device_parameters, run_mode = False)
                break
            except:
                pass 
            time.sleep(0.002)
        
        self.layers = layers
        SIMsalabim_params  = {}

        for layer in layers:
            SIMsalabim_params[layer[1]] = ReadParameterFile(os.path.join(session_path,layer[2]))

        self.SIMsalabim_params = SIMsalabim_params
        pnames = list(SIMsalabim_params[list(SIMsalabim_params.keys())[0]].keys())
        pnames = pnames + list(SIMsalabim_params[list(SIMsalabim_params.keys())[1]].keys())
        self.pnames = pnames    


    def target_metric(self,y,yfit):

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

        # for param in self.params:
            # check is log_scale is a bool
            # if param.log_scale:
            #     parameters[param.name] = 10**parameters[param.name]

        yfit = self.run(parameters)
        y = self.y
       
        # metric = self.target_metric(y,yfit)
        # metric = self.loss_function(metric)

        # train.report({self.metric: loss_function(self.target_metric(y,yfit),loss=self.loss)})

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
    
    def ambi_param_transform(self, param, value, cmd_pars):
        if '.' in param.name:
            layer, par = param.name.split('.')
            if par == 'N_ions': # this is a special case that defines both N_anion and N_cation as the same value
                if param.value_type == 'float':
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.N_anion', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.N_cation', 'val': str(10**param.value)})
                    else:
                        cmd_pars.append({'par': layer+'.N_anion', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.N_cation', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.N_anion', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.N_cation', 'val': str(value)})
            elif par == 'mu_ions':
                if param.value_type == 'float':
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.mu_anion', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.mu_cation', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.mu_anion', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.mu_cation', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.mu_anion', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.mu_cation', 'val': str(value)})
            elif par == 'mu_np':
                if param.value_type == 'float':
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.mu_n', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.mu_p', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.mu_n', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.mu_p', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.mu_n', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.mu_p', 'val': str(value)})
            elif par == 'C_np_bulk':
                if param.value_type == 'float':
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.C_n_bulk', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.C_p_bulk', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.C_n_bulk', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.C_p_bulk', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.C_n_bulk', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.C_p_bulk', 'val': str(value)})
            elif par == 'C_np_int':
                if param.value_type == 'float':
                    if param.force_log:
                        cmd_pars.append({'par': layer+'.C_n_int', 'val': str(10**value)})
                        cmd_pars.append({'par': layer+'.C_p_int', 'val': str(10**value)})
                    else:
                        cmd_pars.append({'par': layer+'.C_n_int', 'val': str(value*param.fscale)})
                        cmd_pars.append({'par': layer+'.C_p_int', 'val': str(value*param.fscale)})
                else:
                    cmd_pars.append({'par': layer+'.C_n_int', 'val': str(value)})
                    cmd_pars.append({'par': layer+'.C_p_int', 'val': str(value)})
        
        return cmd_pars

    def energy_level_offsets(self, custom_pars, clean_pars):
        
        # make a deepcopy of self.SIMsalabim_params to avoid mixing the values of the energy levels when running in parallel
        tmp_SIMsalabim_params = copy.deepcopy(self.SIMsalabim_params)

        # search for energy level values defined in clean_pars and add them to the SIMsalabim_params
        for cmd in clean_pars:
            if ('E_c' in cmd['par']) and (not 'offset' in cmd['par']) and (not 'Egap' in cmd['par']):
                layer, par = cmd['par'].split('.')
                tmp_SIMsalabim_params[layer][par] = cmd['val']
            if ('E_v' in cmd['par']) and (not 'offset' in cmd['par']) and (not 'Egap' in cmd['par']):
                layer, par = cmd['par'].split('.')
                tmp_SIMsalabim_params[layer][par] = cmd['val']
 
        Ec_cmd_nrj, Ev_cmd_nrj, Ec_idx_in_stack, Ec_idx_in_cmd_pars, Ev_idx_in_stack, Ev_idx_in_cmd_pars = [],[],[],[],[],[]
        Egap_cmd_nrj, W_L_offset, W_R_offset = [],[],[]
        for idx, cmd in enumerate(custom_pars):
            if '.' in cmd['par'] and 'offset' in cmd['par'] and not 'W_L' in cmd['par'] and not 'W_R' in cmd['par']:
                layer, par = cmd['par'].split('.')
                offset, layer1, layer2 = layer.split('_')
                if par == 'E_c':
                    Ec_idx_in_stack.append(int(layer1[1:]))
                    Ec_idx_in_cmd_pars.append(idx)
                
                if par == 'E_v':
                    Ev_idx_in_stack.append(int(layer1[1:]))
                    Ev_idx_in_cmd_pars.append(idx)

            if '.' in cmd['par'] and 'Egap' in cmd['par']:
                Egap_cmd_nrj.append(cmd)
            
            if '.' in cmd['par'] and 'offset' in cmd['par'] and 'W_L' in cmd['par']:
                W_L_offset.append(cmd)

            if '.' in cmd['par'] and 'offset' in cmd['par'] and 'W_R' in cmd['par']:
                W_R_offset.append(cmd)

        # reoder the Ec and Ev in cmd_pars to match the order in the stack
        dum_array = np.asarray([Ec_idx_in_stack, Ec_idx_in_cmd_pars])
        dum_array = dum_array[:, dum_array[0].argsort()] # sort the array based on the first row
        Ec_cmd_nrj = [custom_pars[dum_array[1][i]] for i in range(len(dum_array[1]))]

        dum_array = np.asarray([Ev_idx_in_stack, Ev_idx_in_cmd_pars])
        dum_array = dum_array[:, dum_array[0].argsort()] # sort the array based on the first row
        Ev_cmd_nrj = [custom_pars[dum_array[1][i]] for i in range(len(dum_array[1]))] 

        # Set the energy levels of the layers
        Ec_cmd_nrj = Ec_cmd_nrj[::-1] #  invert order of cmd_nrj
        for idx, cmd in enumerate(Ec_cmd_nrj):
            layer, par = cmd['par'].split('.')
            offset, layer1, layer2 = layer.split('_')
            if int(layer1[1:]) <= int(layer2[1:]):
                raise ValueError('The energy level offset between conduction bands must be define from right to left so the offset should be defined as offset_'+layer2+'_offset_'+layer1+' instead of offset_'+layer1+'_offset_'+layer2)
            if par == 'E_c':
                Ec_val = float(tmp_SIMsalabim_params[layer1]['E_c']) - float(cmd['val'])
                clean_pars.append({'par': layer2+'.E_c', 'val': str(Ec_val)})
                tmp_SIMsalabim_params[layer2]['E_c'] = str(Ec_val)
        
        for idx, cmd in enumerate(Ev_cmd_nrj):
            layer, par = cmd['par'].split('.')
            offset, layer1, layer2 = layer.split('_')
            if int(layer1[1:]) >= int(layer2[1:]):
                raise ValueError('The energy level offset between valence bands must be define from left to right so the offset should be defined as offset_'+layer1+'_offset_'+layer2+' instead of offset_'+layer2+'_offset_'+layer1)
            if par == 'E_v':
                Ev_val = float(tmp_SIMsalabim_params[layer1]['E_v']) - float(cmd['val'])
                clean_pars.append({'par': layer2+'.E_v', 'val': str(Ev_val)})
                tmp_SIMsalabim_params[layer2]['E_v'] = str(Ev_val)

        # Set the bandgap energy level of the layers
        for idx, cmd in enumerate(Egap_cmd_nrj):
            layer, par = cmd['par'].split('.')
            Egap, layer_ = layer.split('_')
            if par == 'E_c':
                E_v = float(tmp_SIMsalabim_params[layer_]['E_v'])
                E_c = E_v - float(cmd['val'])
                clean_pars.append({'par': layer_+'.E_c', 'val': str(E_c)})
                tmp_SIMsalabim_params[layer_]['E_c'] = str(E_c)
            if par == 'E_v':
                E_c = float(tmp_SIMsalabim_params[layer_]['E_c'])
                E_v = E_c + float(cmd['val'])
                clean_pars.append({'par': layer_+'.E_v', 'val': str(E_v)})
                tmp_SIMsalabim_params[layer_]['E_v'] = str(E_v)

        # finish with the electrode offsets
        for idx, cmd in enumerate(W_L_offset):
            layer, par = cmd['par'].split('.')
            if par == 'E_c':
                if float(cmd['val']) > 0:
                    raise ValueError('The offset of the work function of the left electrode with respect to the conduction band must be negative')
                W_L = float(tmp_SIMsalabim_params['l1']['E_c']) - float(cmd['val'])
                clean_pars.append({'par': 'W_L', 'val': str(W_L)})
                tmp_SIMsalabim_params['setup']['W_L'] = str(W_L)
            if par == 'E_v':
                if float(cmd['val']) < 0:
                    raise ValueError('The offset of the work function of the left electrode with respect to the valence band must be positive')
                W_L = float(tmp_SIMsalabim_params['l1']['E_v']) - float(cmd['val'])
                clean_pars.append({'par': 'W_L', 'val': str(W_L)})
                tmp_SIMsalabim_params['setup']['W_L'] = str(W_L)
        
        for idx, cmd in enumerate(W_R_offset):
            layer, par = cmd['par'].split('.')
            keys_list = list(tmp_SIMsalabim_params.keys())
            last_layer = keys_list[-1]
            if par == 'E_c':
                if float(cmd['val']) > 0:
                    raise ValueError('The offset of the work function of the right electrode with respect to the conduction band must be negative')
                W_R = float(tmp_SIMsalabim_params[last_layer]['E_c']) - float(cmd['val'])
                clean_pars.append({'par': 'W_R', 'val': str(W_R)})
                tmp_SIMsalabim_params['l1']['W_R'] = str(W_R)
            if par == 'E_v':
                if float(cmd['val']) < 0:
                    raise ValueError('The offset of the work function of the right electrode with respect to the valence band must be positive')
                W_R = float(tmp_SIMsalabim_params[last_layer ]['E_v']) - float(cmd['val'])
                clean_pars.append({'par': 'W_R', 'val': str(W_R)})
                tmp_SIMsalabim_params['setup']['W_R'] = str(W_R)

        return clean_pars    
                
    def check_duplicated_parameters(self, cmd_pars):

        names = []
        for cmd in cmd_pars:
            if cmd['par'] in names:
                raise ValueError('Parameter '+cmd['par']+' is defined more than once in the cmd_pars. Please remove the duplicates.')
            names.append(cmd['par'])

    def prepare_cmd_pars(self, parameters, custom_pars, clean_pars,VarNames):

        for param in self.params:
            if param.name in parameters.keys():
                if param.name not in VarNames:
                    VarNames.append(param.name)
                    if '.' in param.name and 'offset' not in param.name and 'Egap' not in param.name:
                        layer, par = param.name.split('.')
                        if par not in ['Nions', 'mu_ions', 'mu_np', 'C_np_bulk', 'C_np_int']:
                            if par in self.SIMsalabim_params[layer].keys():
                                if param.value_type == 'float':
                                    if param.force_log:
                                        clean_pars.append({'par': param.name, 'val': str(10**parameters[param.name])})
                                    else:
                                        clean_pars.append({'par': param.name, 'val': str(parameters[param.name]*param.fscale)})
                                else:
                                    clean_pars.append({'par': param.name, 'val': str(parameters[param.name])})
                            else:
                                # put in custom_pars
                                if param.value_type == 'float':
                                    if param.force_log:
                                        custom_pars.append({'par': param.name, 'val': str(10**parameters[param.name])})
                                    else:
                                        custom_pars.append({'par': param.name, 'val': str(parameters[param.name]*param.fscale)})
                                else:
                                    custom_pars.append({'par': param.name, 'val': str(parameters[param.name])})
                        else:
                            clean_pars = self.ambi_param_transform(param, parameters[param.name], clean_pars)                      
                    else:
                        if param.name in self.SIMsalabim_params['setup'].keys():
                            if param.value_type == 'float':
                                if param.force_log:
                                    clean_pars.append({'par': param.name, 'val': str(10**parameters[param.name])})
                                else:
                                    clean_pars.append({'par': param.name, 'val': str(parameters[param.name]*param.fscale)})
                            else:
                                clean_pars.append({'par': param.name, 'val': str(parameters[param.name])})
                        else:
                            # put in custom_pars
                            if 'offset' in param.name or 'Egap' in param.name:
                                if param.value_type == 'float':
                                    if param.force_log:
                                        custom_pars.append({'par': param.name, 'val': str(10**parameters[param.name])})
                                    else:
                                        custom_pars.append({'par': param.name, 'val': str(parameters[param.name]*param.fscale)})
                                else:
                                    custom_pars.append({'par': param.name, 'val': str(parameters[param.name])})
                            else:
                                warnings.warn('Parameter '+param.name+' is not defined in the SIMsalabim parameter files. Please check the parameter names. The optimization will proceed but '+param.name+' will not be used by SIMsalabim.', UserWarning)
                            # raise ValueError('Parameter '+param.name+' is not defined in the SIMsalabim parameter files. Please check the parameter names.')
                        
                else:
                    raise ValueError('Parameter '+param.name+' is defined in both the parameters and cmd_pars. Please remove one of them.')
            else:
                raise ValueError('There is no parameter named '+param.name+' in the self.params list. Please check the parameter names.')

        return custom_pars, clean_pars, VarNames


    def run(self, parameters):
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
        UUID = str(uuid.uuid4())

        ret, mess = run_SS_JV(self.simss_device_parameters, self.session_path, JV_file_name = 'JV.dat', G_fracs = Gfracs, UUID=UUID, cmd_pars=clean_pars, parallel = parallel, max_jobs = max_jobs)
 
        if type(ret) == int:
            if not ret == 0 :
                return np.nan

        elif isinstance(ret, subprocess.CompletedProcess):
            
            if not(ret.returncode == 0 or ret.returncode == 95):
                return np.nan

        else:
            if not all([(res == 0 or res == 95) for res in ret]):
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

            
