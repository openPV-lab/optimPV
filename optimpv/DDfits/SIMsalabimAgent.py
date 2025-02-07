"""Provides general functionality for Agent objects for SIMsalabim simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy
from scipy import interpolate

from optimpv import *
from optimpv.general.general import calc_metric, loss_function
from optimpv.general.BaseAgent import BaseAgent
from pySIMsalabim import *

######### Agent Definition #######################################################################
class SIMsalabimAgent(BaseAgent):
    """ Provides general functionality for Agent objects for SIMsalabim simulations
    
    """
    
    def __init__(self) -> None:
        pass
        

    def get_SIMsalabim_clean_cmd(self, parameters, sim_type='simss'):
        """Get the command line arguments for the SIMsalabim simulation with properly formatted parameters

        Parameters
        ----------
        parameters : dict or list of Fitparam() objects
            dictionary of parameter names and values or list of Fitparam() objects
        sim_type : str, optional
            type of simulation ('simss' or 'zimt'), by default 'simss'

        Returns
        -------
        str
            command line arguments for the SIMsalabim simulation
        """        

        # if parameters is not a dict:
        if not isinstance(parameters, dict):
            dummy_pars = {}
            for param in parameters:
                if param.value_type == 'float':
                    if param.force_log:
                        dummy_pars[param.name] = np.log10(param.value)
                    else:
                        dummy_pars[param.name] = param.value/param.fscale
                elif param.value_type == 'int':
                    dummy_pars[param.name] = param.value/param.stepsize
                else:
                    dummy_pars[parameters.name] = parameters.value
            # parameters = dummy_pars
        else:
            dummy_pars = copy.deepcopy(parameters)
                
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

        custom_pars, clean_pars, VarNames = self.prepare_cmd_pars(dummy_pars, custom_pars, clean_pars, VarNames)

        clean_pars = self.energy_level_offsets(custom_pars, clean_pars)

        self.check_duplicated_parameters(clean_pars)

        return construct_cmd(sim_type, clean_pars)
    

    def package_SIMsalabim_files(self, parameters, sim_type, save_path  = None):
        """ Package the SIMsalabim files for the simulation

        Parameters
        ----------
        parameters : dict or list of Fitparam() objects
            dictionary of parameter names and values or list of Fitparam() objects
        sim_type : str
            type of simulation ('simss' or 'zimt')
        save_path : str, optional
            path to save the simulation files, if None, it will be saved in the session_path/tmp_results, by default None

        Raises
        ------
        ValueError
            if sim_type is not 'simss' or 'zimt'
        """        

        # if parameters is not a dict:
        if not isinstance(parameters, dict):
            dummy_pars = {}
            for param in parameters:
                if param.value_type == 'float':
                    dummy_pars[param.name] = param.value/param.fscale
                elif param.value_type == 'int':
                    dummy_pars[param.name] = param.value/param.stepsize
                else:
                    dummy_pars[parameters.name] = parameters.value
        else:
            dummy_pars = copy.deepcopy(parameters)
                
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

        custom_pars, clean_pars, VarNames = self.prepare_cmd_pars(dummy_pars, custom_pars, clean_pars, VarNames)

        clean_pars = self.energy_level_offsets(custom_pars, clean_pars)

        dum_dic= {}
        for cmd in clean_pars:
            dum_dic[cmd['par']] = cmd['val']


        if save_path is None:
            save_path = os.path.join(self.session_path,'tmp_results')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # copy sim_type file from session_path to save_path
        if sim_type == 'simss':
            if os.name == 'nt':
                shutil.copy(os.path.join(self.session_path,'simss.exe'),os.path.join(save_path,'simss.exe'))
            else:
                shutil.copy(os.path.join(self.session_path,'simss'),os.path.join(save_path,'simss'))
        elif sim_type == 'zimt':
            if os.name == 'nt':
                shutil.copy(os.path.join(self.session_path,'zimt.exe'),os.path.join(save_path,'zimt.exe'))
            else:
                shutil.copy(os.path.join(self.session_path,'zimt'),os.path.join(save_path,'zimt'))
        else:
            raise ValueError('sim_type must be either simss or zimt')

        # get all input files that are needed for the simulation and copy them to save_path with their basename
        InputFiles2copy = []
        dev_par_new = copy.deepcopy(self.dev_par)
        for layer in self.layers:
            if layer[1] == 'setup':
                for section in dev_par_new[layer[2]]:   
                    # print(section[1:])
                    for param in section[1:]:
                        if param[0] == 'par':
                            if param[1] in dum_dic.keys():
                                if self.is_inputFile(param[1],dum_dic[param[1]]):
                                    if os.path.basename(dum_dic[param[1]]) != 'none':
                                        InputFiles2copy.append(dum_dic[param[1]])
                                    param[2] = os.path.basename(dum_dic[param[1]])
                                else:
                                    param[2] = dum_dic[param[1]]
                            else:
                                if self.is_inputFile(param[1],param[2]):
                                    if os.path.basename(param[2]) != 'none':
                                        InputFiles2copy.append(param[2])
                                    param[2] = os.path.basename(param[2])
                                else:
                                    param[2] = param[2]
            else:
                for section in dev_par_new[layer[2]]:
                    for param in section[1:]:
                        if param[0] == 'par':
                            if layer[1]+'.'+param[1] in dum_dic.keys():
                                if self.is_inputFile(param[1],dum_dic[layer[1]+'.'+param[1]]):
                                    if os.path.basename(dum_dic[layer[1]+'.'+param[1]]) != 'none':
                                        InputFiles2copy.append(dum_dic[layer[1]+'.'+param[1]])
                                    param[2] = self.convert_parameter_to_basename(param[1],dum_dic[layer[1]+'.'+param[1]])
                                else:
                                    param[2] = dum_dic[layer[1]+'.'+param[1]]
                            else:
                                if self.is_inputFile(param[1],param[2]):
                                    if os.path.basename(param[2]) != 'none':
                                        InputFiles2copy.append(param[2])
                                    param[2] = self.convert_parameter_to_basename(param[1],param[2])
                                else:
                                    param[2] = param[2]

        #copy the simulation setup file
        shutil.copy(os.path.join(self.session_path,self.simulation_setup),os.path.join(save_path,os.path.basename(self.simulation_setup)))
        for file in InputFiles2copy:
            if os.path.isfile(os.path.join(self.session_path,file)):
                shutil.copy(os.path.join(self.session_path,file),os.path.join(save_path,os.path.basename(file)))

        #update keys with new filenames by taking the basename
        dum_dic = {}
        for key in dev_par_new.keys():
            dum_dic[os.path.basename(key)] = dev_par_new[key]
        dev_par_new = dum_dic

        keys = list(dev_par_new.keys())

        for key in dev_par_new.keys():
            with open(os.path.join(save_path,key),'w') as f:
                f.write(devpar_write_to_txt(dev_par_new[key]))

    def ambi_param_transform(self, param, value, cmd_pars):
        """Transform the ambipolar parameters to the SIMsalabim format  
        example:  
        mu_ions -> mu_anion and mu_cation  
        N_ions -> N_anion and N_cation  
        mu_np -> mu_n and mu_p  
        C_np_bulk -> C_n_bulk and C_p_bulk  
        C_np_int -> C_n_int and C_p_int  

        Parameters
        ----------
        param : Fitparam() object
            Fitparam() object
        value : float
            value of the parameter
        cmd_pars : list of dict
            list of dictionaries with the following form {'par': string, 'val': string} 

        Returns
        -------
        list of dict
            list of dictionaries with the following form {'par': string, 'val': string} 
        """        
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
        """Convert the energy level offsets to the SIMsalabim format energy levels

        Parameters
        ----------
        custom_pars : list of dict
            list of dictionaries these contain all the parameters that are not explicitely in the SIMsalabim format and not in the ambipolar format (see ambi_param_transform). The dictionaries are of the form {'par': string, 'val': string}  
        clean_pars : list of dict
            list of dictionaries these contain all the parameters that are explicitely in the SIMsalabim format. The dictionaries are of the form {'par': string, 'val': string}

        Returns
        -------
        list of dict
            list of dictionaries these contain all the parameters converted to the SIMsalabim format. The dictionaries are of the form {'par': string, 'val': string}

        Raises
        ------
        ValueError
            Energy level offset between conduction bands must be defined from right to left
        ValueError
            Energy level offset between valence bands must be defined from left to right
        ValueError
            The offset of the work function of the left electrode with respect to the conduction band must be negative
        ValueError
            The offset of the work function of the left electrode with respect to the valence band must be positive
        ValueError
            The offset of the work function of the right electrode with respect to the conduction band must be negative
        ValueError
            The offset of the work function of the right electrode with respect to the valence band must be positive
        """        
        
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
        """Check if there are duplicated parameters in the cmd_pars

        Parameters
        ----------
        cmd_pars : list of dict
            list of dictionaries with the following form {'par': string, 'val': string}

        Raises
        ------
        ValueError
            There are duplicated parameters in the cmd_pars
        """        
        names = []
        for cmd in cmd_pars:
            if cmd['par'] in names:
                raise ValueError('Parameter '+cmd['par']+' is defined more than once in the cmd_pars. Please remove the duplicates.')
            names.append(cmd['par'])

    def prepare_cmd_pars(self, parameters, custom_pars, clean_pars,VarNames):
        """Prepare the cmd_pars for the SIMsalabim simulation

        Parameters
        ----------
        parameters : dict
            dictionary of parameter names and values
        custom_pars : list of dict
            list of dictionaries containing the custom parameters that do not have a direct match in the SIMsalabim parameter files with the following form {'par': string, 'val': string}
        clean_pars : list of dict
            list of dictionaries containing the parameters that have a direct match in the SIMsalabim parameter files with the following form {'par': string, 'val': string}
        VarNames : list of str
            list of parameter names

        Returns
        -------
        list of dict
            list of dictionaries containing the custom parameters that do not have a direct match in the SIMsalabim parameter files with the following form {'par': string, 'val': string}
        list of dict
            list of dictionaries containing the parameters that have a direct match in the SIMsalabim parameter files with the following form {'par': string, 'val': string}
        list of str
            list of parameter names

        Raises
        ------
        ValueError
            If the parameter name is not in the self.params list
        ValueError
            If the parameter name is in both the parameters and cmd_pars
        """        

        for param in self.params:
            if param.name in parameters.keys():
                if param.name not in VarNames:
                    VarNames.append(param.name)
                    if '.' in param.name and 'offset' not in param.name and 'Egap' not in param.name:
                        layer, par = param.name.split('.')
                        if par not in ['N_ions', 'mu_ions', 'mu_np', 'C_np_bulk', 'C_np_int']:
                            if par in self.SIMsalabim_params[layer].keys():
                                if param.value_type == 'float':
                                    if param.force_log:
                                        clean_pars.append({'par': param.name, 'val': str(10**parameters[param.name])})
                                    else:
                                        clean_pars.append({'par': param.name, 'val': str(parameters[param.name]*param.fscale)})
                                elif param.value_type == 'int':
                                    clean_pars.append({'par': param.name, 'val': str(int(parameters[param.name]*param.stepsize))})
                                else:
                                    clean_pars.append({'par': param.name, 'val': str(parameters[param.name])})
                            else:
                                # put in custom_pars
                                if param.value_type == 'float':
                                    if param.force_log:
                                        custom_pars.append({'par': param.name, 'val': str(10**parameters[param.name])})
                                    else:
                                        custom_pars.append({'par': param.name, 'val': str(parameters[param.name]*param.fscale)})
                                elif param.value_type == 'int':
                                    custom_pars.append({'par': param.name, 'val': str(int(parameters[param.name]*param.stepsize))})
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
                            elif param.value_type == 'int':
                                clean_pars.append({'par': param.name, 'val': str(int(parameters[param.name]*param.stepsize))})
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
                                elif param.value_type == 'int':
                                    custom_pars.append({'par': param.name, 'val': str(int(parameters[param.name]*param.stepsize))})
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
    
    def convert_parameter_to_basename(self, name, value):
        """Convert the parameter value to its basename if it is a file

        Parameters
        ----------
        name : str
            parameter name
        value : str or float or int
            parameter value

        Returns
        -------
        str
            parameter value in its basename
        """        

        if name.endswith('File'):
            return os.path.basename(value)
        
        elif name.startswith('l') and name[1:].isdigit():
            return os.path.basename(value)
        
        elif name == 'genProfile':
            if value.lower() != 'calc' and value.lower() != 'none':
                return os.path.basename(value)
        
        elif name.lower() == 'expjv' and value.lower() != 'none':
            return os.path.basename(value)
        
        elif name.startswith('nk'):
            return os.path.basename(value)
        
        elif name.lower() == 'spectrum':
            return os.path.basename(value)
        
        else:
            return value

    def is_inputFile(self, name, value):
        """Check if the parameter is an input file

        Parameters
        ----------
        name : str
            parameter name
        value : str or float or int
            parameter value

        Returns
        -------
        bool
            True if the parameter is an input file, False otherwise
        """        
        if name.endswith('File'):
            return True
        elif name.startswith('l') and name[1:].isdigit():
            return True
        elif name == 'genProfile':
            if value.lower() != 'calc' and value.lower() != 'none':
                return True
        elif name.lower() == 'expjv' and value.lower() != 'none':
            return True
        elif name.startswith('nk'):
            return True
        elif name.lower() == 'spectrum':
            return True
        else:
            return False    