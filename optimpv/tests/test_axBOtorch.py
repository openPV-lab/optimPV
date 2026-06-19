""" Test EQE module with pySIMsalabim"""

######### Package Imports #########################################################################

import os, sys, torch, copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    from optimpv import *
    from optimpv.optimizers.axBOtorch.axUtils import *
    from optimpv.models.Diodefits.DiodeAgent import DiodeAgent
    from optimpv.models.Diodefits.DiodeModel import *
except Exception as e:
    # Add the parent directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from optimpv import *
    from optimpv.optimizers.axBOtorch.axUtils import *
    from optimpv.models.Diodefits.DiodeAgent import DiodeAgent
    from optimpv.models.Diodefits.DiodeModel import *

from optimpv.models.RateEqfits.RateEqAgent import RateEqAgent
from optimpv.models.RateEqfits.RateEqModel import *
from optimpv.models.RateEqfits.Pumps import *
from optimpv.optimizers.axBOtorch.axBOtorchOptimizer import axBOtorchOptimizer
from optimpv.general.SuggestOnlyAgent import SuggestOnlyAgent

from botorch.acquisition.logei import qLogNoisyExpectedImprovement 
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement   
from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from ax.adapter.transforms.standardize_y import StandardizeY
from ax.adapter.transforms.unit_x import UnitX
from ax.adapter.transforms.remove_fixed import RemoveFixed
from ax.adapter.transforms.log import Log
from ax.generators.torch.botorch_modular.utils import ModelConfig
from ax.generators.torch.botorch_modular.surrogate import SurrogateSpec
from gpytorch.kernels import MaternKernel
from gpytorch.kernels import ScaleKernel


######### Test Functions #########################################################################

def test_SOO_diode_fit():
    """Test the single-objective optimization of a diode model using axBOtorchOptimizer."""
    try:
        params = []

        J0 = FitParam(name = 'J0', value = 1e-5, bounds = [1e-6,1e-3], log_scale = True, rescale = False, value_type = 'float', type='range', display_name=r'$J_0$', unit='A m$^{-2}$', axis_type = 'log',force_log=True)
        params.append(J0)

        n = FitParam(name = 'n', value = 1.5, bounds = [1,2], log_scale = False, value_type = 'float', type='range', display_name=r'$n$', unit='', axis_type = 'linear')
        params.append(n)

        R_series = FitParam(name = 'R_series', value = 1e-4, bounds = [1e-5,1e-3], log_scale = True, rescale = False, value_type = 'float', type='range', display_name=r'$R_{\text{series}}$', unit=r'$\Omega$ m$^2$', axis_type = 'log',force_log=True)
        params.append(R_series)

        R_shunt = FitParam(name = 'R_shunt', value = 1e-1, bounds = [1e-2,1e2], log_scale = True, rescale = False, value_type = 'float', type='range', display_name=r'$R_{\text{shunt}}$', unit=r'$\Omega$ m$^2$', axis_type = 'log',force_log=True)
        params.append(R_shunt)

        # original values
        params_orig = copy.deepcopy(params)

        # Create JV to fit
        X = np.linspace(0.001,1,100)
        y = NonIdealDiode_dark(X, J0.value, n.value, R_series.value, R_shunt.value)

        # Define the Agent and the target metric/loss function
        metric = 'mse' # can be 'nrmse', 'mse', 'mae'
        loss = 'soft_l1' # can be 'linear', 'huber', 'soft_l1'
        exp_format = 'dark' # can be 'dark', 'light' depending on the type of data you have
        use_pvlib = False # if True, use pvlib to calculate the diode model if not use the implementation in DiodeModel.py

        diode = DiodeAgent(params, X, y, metric = metric, loss = loss, minimize=True,exp_format=exp_format,use_pvlib=use_pvlib,transforms='log')

        model_kwargs_list = [{},{"torch_device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),'botorch_acqf_class':qLogNoisyExpectedImprovement,'transforms':[RemoveFixed, Log,UnitX, StandardizeY],'surrogate_spec':SurrogateSpec(model_configs=[ModelConfig(botorch_model_class=SingleTaskGP,covar_module_class=ScaleKernel, covar_module_options={'base_kernel':MaternKernel(nu=2.5, ard_num_dims=len(params))})])}]

        # Define the optimizer
        optimizer = axBOtorchOptimizer(params = params, agents = diode, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,10], batch_size = [10,2], ax_client = None,  max_parallelism = -1, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = None, name = 'ax_opti',parallel_agents=True,verbose_logging=False)

        optimizer.optimize() # run the optimization with ax


        # get the best parameters and update the params list in the optimizer and the agent
        ax_client = optimizer.ax_client # get the ax client
        optimizer.update_params_with_best_balance() # update the params list in the optimizer with the best parameters
        diode.params = optimizer.params # update the params list in the agent with the best parameters
        assert True

    except Exception as e:
        assert False, "Error occurred during diode fitting: {}".format(e)

def test_MOO_trPLtrMC():
    """Test the multi-objective optimization of a rate equation model using axBOtorchOptimizer."""

    # Define the parameters for the rate equation model
    try:
        params = []

        k_direct = FitParam(name = 'k_direct', value = 3.9e-17, bounds = [1e-18,1e-16], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{direct}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(k_direct)

        k_trap = FitParam(name = 'k_trap', value = 4e-18, bounds = [1e-19,1e-17], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{trap}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(k_trap)

        k_detrap = FitParam(name = 'k_detrap', value = 3.1e-18, bounds = [1e-19,1e-17], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{detrap}}$', unit='s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(k_detrap)

        N_t_bulk = FitParam(name = 'N_t_bulk', value = 1.6e23, bounds = [1e22,5e23], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$N_{\text{t,bulk}}$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
        params.append(N_t_bulk)

        I_factor_PL = FitParam(name = 'I_factor_PL', value = 1e-32, bounds = [1e-33,1e-31], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$I_{\text{PL}}$', unit='-', axis_type = 'log', force_log=True)
        params.append(I_factor_PL)

        I_factor_MC = FitParam(name = 'I_factor_MC', value = 2.2e-26, bounds = [1e-27,1e-25], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$I_{\text{PL}}$', unit='-', axis_type = 'log', force_log=True)
        params.append(I_factor_MC)

        ratio_mu = FitParam(name = 'ratio_mu', value = 4.2, bounds = [1,10], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$\mu_{\text{ratio}}$', unit='-', axis_type = 'linear', force_log=False)
        params.append(ratio_mu)

        # Define the path to the data 
        curr_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        path2data  = os.path.join(parent_dir,'Data','perovskite_trPL_trMC')
        filenames = ['S25D1_L532_F0.csv','S25D1_L532_F1.csv','S25D1_L532_F2.csv'] # list of filenames to be analyzed
        res_dir = os.path.join(parent_dir,'temp') # path to the results directory

        # Select Gfracs used for the data
        Gfracs = [1, 0.552, 0.290, 0.136, 0.087]

        # Create a class that contains to do some basic data processing on the data
        class Experiment:
            """ A set of measurements """
            def __init__(self, path2data, filenames, Gfracs, laserCenter=0, num_pts=1e3, take_log=False):
                self.path2data = path2data
                self.filenames = filenames
                self.Gfracs = Gfracs
                self.laserCenter = laserCenter
                self.num_pts = num_pts
                self.take_log = take_log
                
                self.get_data()
                pass
            
            
            def get_data(self):
                self.X_raw, self.y_raw_MW, self.y_raw_PL = [],[],[]
                for filename in self.filenames:
                    # Create empty lists to store data
                    X,y_MW, y_PL = [],[],[]
                    
                    #Load file and extract data
                    with open(os.path.join(self.path2data, filename)) as f:
                        for line in f:
                            tmp=line.strip("\n").split(",")
                            X.append(float(tmp[3]))
                            y_MW.append(float(tmp[4]))
                            
                            if len(tmp)>8:
                                y_PL.append(float(tmp[10]))
                            else:
                                raise ValueError("The file does not contain PL data")
                    
                    # Create output arrays
                    self.X_raw.append(np.array(X))
                    self.y_raw_MW.append(np.array(y_MW))
                    self.y_raw_PL.append(np.array(y_PL))
                pass
            
            
            def process_data(self, cut_rise=False, cut_time=None, cut_sigma=False):
                # Create empty lists to store data
                X_out, y_out_MW, y_out_PL = [],[],[]
                self.X_processed, self.y_processed_MW, self.y_processed_PL = [],[],[]
                self.signalParams={}
                self.background_out_PL=[]
                # Data processing:
                for X, y_MW, y_PL, Gfrac in zip(self.X_raw, self.y_raw_MW, self.y_raw_PL, self.Gfracs):
                        
                    # Subtract the background from MW and PL data
                    index = np.where(X<(-10e-9))  # Calculate the background from the average of the signal up to 10ns before the peak (this buffer is to prevent the rise of the peak to affect the background)
                    self.signalParams["MW_background"] = np.mean(y_MW[index])
                    self.signalParams["PL_background"] = np.mean(y_PL[index])
                    self.signalParams["MW_sigma"] = np.std(y_MW[index])
                    self.signalParams["PL_sigma"] = np.std(y_PL[index])
                    
                    
                    y_MW = y_MW - self.signalParams["MW_background"]
                    y_PL = y_PL - self.signalParams["PL_background"]
                    print('PL Sigma {}, PL background {}, MW Sigma {}, MW background {}'.format(self.signalParams["PL_sigma"],self.signalParams["PL_background"],self.signalParams["MW_sigma"],self.signalParams["MW_background"]))
                    
                    # Find the peak position
                    self.signalParams["index_max_MW"] = np.argmax(abs(y_MW))
                    self.signalParams["index_max_PL"] = np.argmax(abs(y_PL))
                    
                    # Find the sign of the peak
                    self.signalParams["sign_max_MW"] = np.sign(y_MW[self.signalParams["index_max_MW"]])
                    self.signalParams["sign_max_PL"] = np.sign(y_PL[self.signalParams["index_max_PL"]])
                    
                    # Remove datapoints at the beginning of the signal
                    if cut_rise == "MW":
                        index = np.where(X >= X[self.signalParams["index_max_MW"]])
                    elif cut_rise == "PL":
                        index = np.where(X >= X[self.signalParams["index_max_PL"]])
                    elif cut_rise == "Time":
                        index = np.where(X > cut_time)
                    else:
                        index = np.where(X > self.laserCenter)
                        
                    X = X[index]
                    # Remove datapoints before the laser peak from the MW and PL signal and make sure, that the peak is positive
                    y_MW = y_MW[index]*self.signalParams["sign_max_MW"]
                    y_PL = y_PL[index]*self.signalParams["sign_max_PL"]
                    
                    # Remove datapoints that aren't significant enough (in either measurement)
                    if cut_sigma:
                        sigma = float(cut_sigma)
                        index = np.where((np.abs(y_MW)>sigma*self.signalParams["MW_sigma"]) & (np.abs(y_PL)>sigma*self.signalParams["PL_sigma"]))

                        X = X[index]
                        y_MW = y_MW[index]
                        y_PL = y_PL[index]
                    
                    
                    # Interpolate to get num_pts
                    X_interp = np.geomspace(X[1],X[-1],int(self.num_pts))

                    # Add 0 to the beginning of X_interp
                    X_interp = np.insert(X_interp,0,0)
                    y_interp_MW = np.interp(X_interp,X,y_MW)
                    y_interp_PL = np.interp(X_interp,X,y_PL)

                    # Take the log of the data
                    if self.take_log:
                        y_interp_MW = np.log10(y_interp_MW)
                        y_interp_PL = np.log10(y_interp_PL)
                        
                        # Remove all data points where either signal is NaN
                        mask_NaNs = np.logical_or(np.isnan(y_interp_PL), np.isnan(y_interp_MW))
                        X_interp = X_interp[~mask_NaNs]
                        y_interp_MW = y_interp_MW[~mask_NaNs]
                        y_interp_PL = y_interp_PL[~mask_NaNs]
                        print('Removed {} Data Points while taking the logarithm!'.format(np.count_nonzero(mask_NaNs)))
                    
                    # Append the data to the output
                    for i in range(len(X_interp)):
                        X_out.append([X_interp[i],Gfrac])
                        y_out_MW.append(y_interp_MW[i])
                        y_out_PL.append(y_interp_PL[i])
                        self.background_out_PL.append(self.signalParams["PL_sigma"]*np.sqrt(2/np.pi))
                        
                    self.X_processed.append(np.array(X_interp))
                    self.y_processed_MW.append(np.array(y_interp_MW))
                    self.y_processed_PL.append(np.array(y_interp_PL))

                # Convert the output to arrays
                self.X = np.array(X_out)
                self.y_MW = np.array(y_out_MW)
                self.y_PL = np.array(y_out_PL)
                pass
        
        # Load the data and process it
        data_exp = Experiment(path2data, filenames, Gfracs, laserCenter=2.8E-8, take_log=False)
        data_exp.process_data(cut_rise=False, cut_time=None ,cut_sigma=0)
        X = data_exp.X
        y_MW = data_exp.y_MW
        y_PL = data_exp.y_PL
        back_PL = data_exp.background_out_PL

        # remove all point where PL is below 0
        mask = np.where(y_PL<0)
        y_PL = np.delete(y_PL,mask)
        X = np.delete(X,mask,axis=0)
        y_MW = np.delete(y_MW,mask)
        back_PL = np.delete(back_PL,mask)
        # remove all point where MW is below 0
        mask = np.where(y_MW<0)
        y_MW = np.delete(y_MW,mask)
        X = np.delete(X,mask,axis=0)
        y_PL = np.delete(y_PL,mask)
        back_PL = np.delete(back_PL,mask)
        from sklearn.preprocessing import minmax_scale
        # Assign weights based on the signal strength
        weight_PL = None #1/(np.abs(y_PL))
        weight_MW = None #1/(np.abs(y_MW))
        # weight_MW = minmax_scale(weight_MW, feature_range=(1,1000))

        # RateEqModel parameters
        fpu = 1e3 # Frequency of the pump laser in Hz
        N0 = 1.041e24 # Initial carrier density in m-3
        background = 0 # Background illumination 
        Gfracs = [1, 0.552, 0.290] # Gfracs used for the data

        # Define the Agent and the target metric/loss function
        metric = 'mse'
        loss = 'soft_l1'
        pump_args = {'N0': N0, 'fpu': fpu , 'background' : background, }

        # 50 log spaced points data X, y_PL, y_MW
        # t_min except 0
        num_pts = 20
        t_min = X[X[:,0]>0,0].min()
        X_log = np.geomspace(t_min,X[:,0].max(),num_pts)
        X_log = np.insert(X_log,0,0)
        # get teh closest 50 points to the log spaced points
        X_50 = np.zeros((int(len(X_log)*len(Gfracs)),2))
        y_PL_50 = np.zeros(int(len(X_log)*len(Gfracs)))
        y_MW_50 = np.zeros(int(len(X_log)*len(Gfracs)))

        idx_main = 0
        for g in Gfracs:
            idx = 0
            for i in range(len(X_log)):
                index = np.argmin(abs(X[X[:,1]==g,0]-X_log[idx]))
                X_50[idx_main] = X[X[:,1]==g][index]
                y_PL_50[idx_main] = y_PL[X[:,1]==g][index]
                y_MW_50[idx_main] = y_MW[X[:,1]==g][index]
                idx += 1
                idx_main += 1
                
        RateEq = RateEqAgent(params, [X,X], [y_PL,y_MW], model = BTD_model, pump_model = initial_carrier_density, pump_args = pump_args, fixed_model_args = {}, metric = [metric,metric], loss = [loss,loss], threshold=[0.5,0.5],minimize=[True,True],exp_format=['trPL','trMC'],detection_limit=1e-5, weight=[weight_PL,weight_MW], transforms ='log')

        model_gen_kwargs_list = None
        parameter_constraints = None
        model_kwargs_list = None

        optimizer = axBOtorchOptimizer(params = params, agents = [RateEq], models = ['CENTER','SOBOL','BOTORCH_MODULAR'],n_batches = [1,1,10], batch_size = [1,10,2], ax_client = None,  max_parallelism = -1, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = None, name = 'ax_opti')

        optimizer.optimize() # run the optimization with ax

        assert True
    except Exception as e:
        assert False, "Error occurred during rate equation fitting: {}".format(e)

def test_SOO_TurBO():
    """Test the single-objective optimization of a diode model using axBOtorchOptimizer."""
    try:
        params = []

        k_direct = FitParam(name = 'k_direct', value = 7.5e-17, bounds = [1e-18,1e-15], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{direct}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(k_direct)

        k_deep = FitParam(name = 'k_deep', value = 1.6e4, bounds = [1e4,1e6], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{deep}}$', unit='s$^{-1}$', axis_type = 'log', force_log=True)
        params.append(k_deep)

        k_c = FitParam(name = 'k_c', value = 1.3e6, bounds = [1e4,1e8], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{c}}$', unit='s$^{-1}$', axis_type = 'log', force_log=True)
        params.append(k_c)

        k_e = FitParam(name = 'k_e', value = 7.5e5, bounds = [1e4,1e8], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{e}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(k_e)

        mu = FitParam(name = 'mu', value = 4e-1*1e-4, bounds = [1e-6,1e-2], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$\mu$', unit='m$^{2}$ V$^{-1}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(mu)

        Sfront = FitParam(name = 'S_front', value = 7*1e-2, bounds = [1e-4,1e-1], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$S_{\text{front}}$', unit='m s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(Sfront)

        Sback = FitParam(name = 'S_back', value = 4*1e-2, bounds = [1e-4,1e-1], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$S_{\text{back}}$', unit='m s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(Sback)

        I_factor_PL = FitParam(name = 'I_factor_PL', value = 1e-28, bounds = [1e-29,1e-27], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$I_{\text{PL}}$', unit='-', axis_type = 'log', force_log=True)
        params.append(I_factor_PL)

        N_A = FitParam(name = 'N_A', value = 2.1e21, bounds = [1e20,1e23], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$N_A$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
        params.append(N_A)

        alpha = FitParam(name = 'alpha', value = 3e8, bounds = [1e6,1e8], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$\alpha$', unit='m$^{-1}$', axis_type = 'log',)
        params.append(alpha)

        L = FitParam(name = 'L', value = 600e-9, bounds = [400e-9,1e-6], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$L$', unit='m', axis_type = 'linear',force_log=True)
        params.append(L)


        # original values
        params_orig = copy.deepcopy(params)
        num_free_params = 0
        dum_dic = {}
        for i in range(len(params)):
            if params[i].force_log:
                dum_dic[params[i].name] = np.log10(params[i].value)
            else:
                dum_dic[params[i].name] = params[i].value/params[i].fscale
        # we need this just to run the model to generate some fake data

            if params[i].type != 'fixed'    :
                num_free_params += 1

        # import Data
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # path to the parent directory if not in Notebooks use os.getcwd()
        path2data  = os.path.join(parent_dir,'Data','perovskite_trPL')
        data_raw = pd.read_csv(os.path.join(path2data,'Seo_FAPI1_Glass_0.dat'), sep=r'\s+',names=['t',"trPL"], skiprows=1)
        # convert time to seconds
        data_raw['t'] = data_raw['t'] * 1e-9 # convert to seconds
        max_idx = data_raw['trPL'].idxmax()
        # remove everything before the maximum
        data_raw = data_raw.iloc[max_idx-1:]
        # reset the index
        data_raw = data_raw.dropna() # remove rows with negative trPL values
        data_raw = data_raw[data_raw['trPL'] >= 0]# remove rows with negative trPL values
        data_raw = data_raw.reset_index(drop=True)

        # interpolate the data to have a logarithmically spaced time axis
        t_log = np.logspace(np.log10(2e-9), np.log10(data_raw['t'].max()), num=1000)
        t_log = np.insert(t_log, 0, 0)  # add 0 to the time array
        # interpolate the trPL values
        trPL_log = np.interp(t_log, data_raw['t'] - data_raw['t'].min(), data_raw['trPL'])
        data2fit = pd.DataFrame({'t': t_log - t_log.min(), 'trPL': trPL_log})

        time = data2fit['t'].values # time in seconds
        X = time
        y = data2fit['trPL'] 
        fpu = 50e3 # Frequency of the pump laser in Hz
        Fluence = 4.8e15 # Fluence in m-2
        z_array = np.linspace(0, L.value, 100) # z-axis in meters
        generation = np.exp(-alpha.value * z_array)
        generation_sum = np.trapezoid(generation, z_array)
        n_0z = Fluence / generation_sum * generation
        N0 = np.mean(n_0z) # mean initial carrier density in m-3
        background = 0e28 # Background illumination 

        metric = 'nrmse'
        loss = 'linear' # 'nrmse' or 'mse' or 'soft_l1' or 'linear'
        pump_args = {'N0': N0, 'fpu': fpu , 'background' : background, }
        exp_format = 'trPL' # experiment format
        RateEq = RateEqAgent(params, [X], [y], model = DBTD_model, pump_model = initial_carrier_density, pump_args = pump_args, fixed_model_args = {}, metric = metric, loss = loss,minimize=True,exp_format=exp_format,detection_limit=0e-5,  transforms ='log')

        model_gen_kwargs_list = None
        parameter_constraints = None

        model_kwargs_list = [{},{"torch_device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),'botorch_acqf_class':qLogNoisyExpectedImprovement,'transforms':[RemoveFixed, Log,UnitX, StandardizeY],'surrogate_spec':SurrogateSpec(model_configs=[ModelConfig(botorch_model_class=SingleTaskGP,covar_module_class=ScaleKernel, covar_module_options={'base_kernel':MaternKernel(nu=2.5, ard_num_dims=num_free_params)})])}]

        optimizer = axBOtorchOptimizer(params = params, agents = RateEq, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,8], batch_size = [8,2], ax_client = None,  max_parallelism = 100, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = model_gen_kwargs_list, name = 'ax_opti',parameter_constraints = parameter_constraints,)

        optimizer.optimize_turbo() # run the optimization with ax

        assert True

    except Exception as e:
        assert False, "Error occurred during diode fitting: {}".format(e)


def test_DOE_SOO():

    try:
        # Define the path to the data 
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        data_dir =os.path.join(parent_dir,'Data','6D_pero_opti') # path to the data directory

        # Load the data
        df = pd.read_csv(os.path.join(data_dir,'6D_pero_opti.csv'),sep=r'\s+') # load the data

        params = [] # list of parameters to be optimized

        Spin_Speed_1 = FitParam(name = 'Spin_Speed_1', value = 1000, bounds = [500,3000], value_type = 'int', display_name='Spin Speed 1', unit='rpm', axis_type = 'linear')
        params.append(Spin_Speed_1)

        Duration_t1 = FitParam(name = 'Duration_t1', value = 10, bounds = [5,35], value_type = 'int', display_name='Duration t1', unit='s', axis_type = 'linear')
        params.append(Duration_t1)

        Spin_Speed_2 = FitParam(name = 'Spin_Speed_2', value = 1000, bounds = [1000,3000], value_type = 'int', display_name='Spin Speed 2', unit='rpm', axis_type = 'linear')
        params.append(Spin_Speed_2)

        Dispense_Speed = FitParam(name = 'Dispense_Speed', value = 100, bounds = [10,400], value_type = 'int', display_name='Dispense Speed', unit='rpm', axis_type = 'linear')
        params.append(Dispense_Speed)

        Duration_t3 = FitParam(name = 'Duration_t3', value = 10, bounds = [5,35], value_type = 'int', display_name='Duration t3', unit='s', axis_type = 'linear')
        params.append(Duration_t3)

        Spin_Speed_3 = FitParam(name = 'Spin_Speed_3', value = 3000, bounds = [2000,5000], value_type = 'int', display_name='Spin Speed 3', unit='rpm', axis_type = 'linear')
        params.append(Spin_Speed_3)

        # Define the Agent and the target metric/loss function
        suggest = SuggestOnlyAgent(params,exp_format='Pmax',minimize=False,tracking_exp_format=['Jsc','Voc','FF'],name=None)

        model_gen_kwargs_list = None
        parameter_constraints = None

        model_kwargs_list = [{"torch_device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),'botorch_acqf_class':qLogNoisyExpectedImprovement,'transforms':[RemoveFixed, Log, UnitX, StandardizeY],'surrogate_spec':SurrogateSpec(model_configs=[ModelConfig(botorch_model_class=SaasFullyBayesianSingleTaskGP,covar_module_class=ScaleKernel, covar_module_options={'base_kernel':MaternKernel(nu=2.5, ard_num_dims=len(params))})])}]
        model_kwargs_list =None

        # Define the optimizer
        optimizer = axBOtorchOptimizer(params = params, agents = suggest, models = ['BOTORCH_MODULAR'],n_batches = [1], batch_size = [6], ax_client = None,  max_parallelism = -1, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = None, name = 'ax_opti',suggest_only = True,existing_data=df,verbose_logging=True)

        optimizer.optimize() # run the optimization with ax

        assert True
    except Exception as e:
        assert False, "Error occurred during DOE fitting: {}".format(e)

def test_DOE_SOO_Turbo():

    try:
        # Define the path to the data 
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        data_dir =os.path.join(parent_dir,'Data','6D_pero_opti') # path to the data directory

        # Load the data
        df = pd.read_csv(os.path.join(data_dir,'6D_pero_opti.csv'),sep=r'\s+') # load the data

        params = [] # list of parameters to be optimized

        Spin_Speed_1 = FitParam(name = 'Spin_Speed_1', value = 1000, bounds = [500,3000], value_type = 'int', display_name='Spin Speed 1', unit='rpm', axis_type = 'linear')
        params.append(Spin_Speed_1)

        Duration_t1 = FitParam(name = 'Duration_t1', value = 10, bounds = [5,35], value_type = 'int', display_name='Duration t1', unit='s', axis_type = 'linear')
        params.append(Duration_t1)

        Spin_Speed_2 = FitParam(name = 'Spin_Speed_2', value = 1000, bounds = [1000,3000], value_type = 'int', display_name='Spin Speed 2', unit='rpm', axis_type = 'linear')
        params.append(Spin_Speed_2)

        Dispense_Speed = FitParam(name = 'Dispense_Speed', value = 100, bounds = [10,400], value_type = 'int', display_name='Dispense Speed', unit='rpm', axis_type = 'linear')
        params.append(Dispense_Speed)

        Duration_t3 = FitParam(name = 'Duration_t3', value = 10, bounds = [5,35], value_type = 'int', display_name='Duration t3', unit='s', axis_type = 'linear')
        params.append(Duration_t3)

        Spin_Speed_3 = FitParam(name = 'Spin_Speed_3', value = 3000, bounds = [2000,5000], value_type = 'int', display_name='Spin Speed 3', unit='rpm', axis_type = 'linear')
        params.append(Spin_Speed_3)

        # Define the Agent and the target metric/loss function
        suggest = SuggestOnlyAgent(params,exp_format='Pmax',minimize=False,tracking_exp_format=['Jsc','Voc','FF'],name=None)

        model_gen_kwargs_list = None
        parameter_constraints = None

        model_kwargs_list = [{"torch_device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),'botorch_acqf_class':qLogNoisyExpectedImprovement,'transforms':[RemoveFixed, Log, UnitX, StandardizeY],'surrogate_spec':SurrogateSpec(model_configs=[ModelConfig(botorch_model_class=SaasFullyBayesianSingleTaskGP,covar_module_class=ScaleKernel, covar_module_options={'base_kernel':MaternKernel(nu=2.5, ard_num_dims=len(params))})])}]
        model_kwargs_list =None

        # Define the optimizer
        optimizer = axBOtorchOptimizer(params = params, agents = suggest, models = ['BOTORCH_MODULAR'],n_batches = [1], batch_size = [6], ax_client = None,  max_parallelism = -1, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = None, name = 'ax_opti',suggest_only = True,existing_data=df,verbose_logging=True)

        best_value_previous_step = 22.973
        kwargs_turbo_state = {'length': 0.4, 'success_counter': 1, 'failure_counter': 1 }
        turbo_state_params = optimizer.optimize_turbo(kwargs_turbo={"best_value": best_value_previous_step}, kwargs_turbo_state=kwargs_turbo_state) # run the optimization with turbo

        assert True
    except Exception as e:
        assert False, "Error occurred during DOE fitting with Turbo: {}".format(e)


def test_DOE_MOO():

    try:
        # Define the path to the data 
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        data_dir =os.path.join(parent_dir,'Data','pero_MOO_opti') # path to the data directory

        # Load the data
        df = pd.read_csv(os.path.join(data_dir,'Pero_PLQY_FWHM.csv'),sep=r',') # load the data

        stepsize_fraction = 0.05
        stepsize_spin_speed = 100

        params = [] # list of parameters to be optimized

        Cs_fraction = FitParam(name = 'Cs_fraction', value = 0, bounds = [0,1], value_type = 'int', display_name='Cs fraction', unit='', axis_type = 'linear',stepsize=stepsize_fraction)
        params.append(Cs_fraction)

        Fa_fraction = FitParam(name = 'Fa_fraction', value = 0, bounds = [0,1], value_type = 'int', display_name='Fa fraction', unit='', axis_type = 'linear',stepsize=stepsize_fraction)
        params.append(Fa_fraction)

        Spin_duration_Antisolvent = FitParam(name = 'Spin_duration_Antisolvent', value = 10, bounds = [5,30], value_type = 'int', display_name='Spin duration Antisolvent', unit='s', axis_type = 'linear')
        params.append(Spin_duration_Antisolvent)

        Spin_duration_High_Speed = FitParam(name = 'Spin_duration_High_Speed', value = 20, bounds = [15,60], value_type = 'int', display_name='Spin duration High Speed', unit='s', axis_type = 'linear')
        params.append(Spin_duration_High_Speed)

        Spin_speed = FitParam(name = 'Spin_speed', value = 1000, bounds = [1000,5000], value_type = 'int', display_name='Spin speed', unit='rpm', axis_type = 'linear',stepsize=stepsize_spin_speed)
        params.append(Spin_speed)

        # Define the Agent and the target metric/loss function
        
        threshold = [0.5, 50] # thresholds for the metrics
        suggest = SuggestOnlyAgent(params,exp_format=['PLQY', 'FWHM'],minimize=[False,True],name=None,threshold=threshold)

        model_gen_kwargs_list = None
        parameter_constraints =[f'{stepsize_fraction}*Cs_fraction + {stepsize_fraction}*Fa_fraction <= 1']

        model_kwargs_list = [{"torch_device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),'botorch_acqf_class':qLogNoisyExpectedImprovement,'transforms':[RemoveFixed, Log, UnitX, StandardizeY],'surrogate_spec':SurrogateSpec(model_configs=[ModelConfig(botorch_model_class=SaasFullyBayesianSingleTaskGP,covar_module_class=ScaleKernel, covar_module_options={'base_kernel':MaternKernel(nu=2.5, ard_num_dims=len(params))})])}]
        model_kwargs_list =None

        # Define the optimizer
        optimizer = axBOtorchOptimizer(params = params, agents = suggest, models = ['BOTORCH_MODULAR'],n_batches = [1], batch_size = [6], ax_client = None,  max_parallelism = -1, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = None, name = 'ax_opti',suggest_only = True,existing_data=df,verbose_logging=True)

        optimizer.optimize() # run the optimization with ax
        assert True
    except Exception as e:
        assert False, "Error occurred during DOE MOO fitting: {}".format(e)


def test_multi_trap():
    try:
        # Define the parameters to be fitted
        params = []

        Eg = FitParam(name = 'Eg', value = 1.553, bounds = [0.5,2.0], log_scale = False, rescale = True, value_type = 'float', type='fixed', display_name=r'$E_g$', unit='eV', axis_type = 'linear')
        params.append(Eg)

        L = FitParam(name = 'L', value = 450e-9, bounds = [400e-9,1e-6], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$L$', unit='m', axis_type = 'linear',force_log=True)
        params.append(L)

        alpha = FitParam(name = 'alpha', value = 64348.30886337494 *1e2, bounds = [1e6,1e8], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$\alpha$', unit='m$^{-1}$', axis_type = 'log',)
        params.append(alpha)

        N_cv = FitParam(name = 'N_cv', value = 2e24, bounds = [1e19,1e25], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$N_{cv}$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
        params.append(N_cv)

        k_direct = FitParam(name = 'k_direct', value = 1.96e-17, bounds = [1e-18,1e-14], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{direct}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(k_direct)

        mu_n = FitParam(name = 'mu_n', value = 1.2e-4, bounds = [1e-6,1e-2], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$\mu_n$', unit='m$^{2}$ V$^{-1}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(mu_n) # 4e-1*1e-4

        mu_p = FitParam(name = 'mu_p', value = 4e-5, bounds = [1e-6,1e-2], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$\mu_p$', unit='m$^{2}$ V$^{-1}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(mu_p) # 4e-1*1e-4

        N_t_bulk_1 = FitParam(name = 'N_t_bulk_1', value = 1.85e23, bounds = [1e19,1e24], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$N_{t,\text{bulk}}$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
        params.append(N_t_bulk_1)

        C_n_1 = FitParam(name = 'C_n_1', value = 4.24e-15, bounds = [1e-19,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{n,1}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(C_n_1)

        C_p_1 = FitParam(name = 'C_p_1', value = 8.85e-19, bounds = [1e-19,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{p,1}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(C_p_1)

        E_t_bulk_1 = FitParam(name = 'E_t_bulk_1', value = 0.2, bounds = [0.05,0.3], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$E_{t,\text{bulk}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=False)
        params.append(E_t_bulk_1)

        N_t_bulk_2 = FitParam(name = 'N_t_bulk_2', value = 1.36e21, bounds = [1e19,1e24], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$N_{t,\text{bulk}}$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
        params.append(N_t_bulk_2)

        C_n_2 = FitParam(name = 'C_n_2', value = 1.72e-13, bounds = [1e-19,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{n,2}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(C_n_2)

        C_p_2 = FitParam(name = 'C_p_2', value = 1.13e-16, bounds = [1e-19,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{p,2}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(C_p_2)

        E_t_bulk_2 = FitParam(name = 'E_t_bulk_2', value = 1.34, bounds = [0.3,Eg.value-0.1], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$E_{t,\text{bulk}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=False)
        params.append(E_t_bulk_2)

        I_factor_PL = FitParam(name = 'I_factor_PL', value = 1.275e-22, bounds = [1e-27,1e-20], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$I_{\text{PL}}$', unit='-', axis_type = 'log', force_log=True)
        params.append(I_factor_PL) # in the following we weill fit the PL with the normalized log transformation so this factor is not useful and can be fixed to any value


        # original values
        params_orig = copy.deepcopy(params)
        num_free_params = 0
        dum_dic = {}
        for i in range(len(params)):
            if params[i].force_log:
                dum_dic[params[i].name] = np.log10(params[i].value)
            else:
                dum_dic[params[i].name] = params[i].value/params[i].fscale
        # we need this just to run the model to generate some fake data

            if params[i].type != 'fixed'    :
                num_free_params += 1

        # Define the path to the data 
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        path2data  = os.path.join(parent_dir,'Data','FAPI_trPL')
        filenames = ['2M11-FAPI27_3600s_10kHz_ND3_1280cps_0.272112uW.dat','1M3-FAPI27_3600s_10kHz_ND2_500cps_0.0737403uW.dat','0M4-FAPI27_3600s_10kHz_ND1_940cps_0.048888uW.dat',] #'3M8-FAPI27_3600s_10kHz_ND5_570cps_1.2156099999999999uW.dat',
        power = []
        pmax = 1.2156099999999999
        pmax = max([float(file.split('_')[-1].replace('uW.dat', '')) for file in filenames])
        for idx, file in enumerate(filenames):
            data_raw = pd.read_csv(os.path.join(path2data,file), sep=r'\s+',names=['idx','t',"trPL"], skiprows=1)
            data_raw = data_raw[['t', 'trPL']]
            # convert time to seconds
            data_raw['t'] = data_raw['t'] * 1e-12 # convert to seconds
            max_idx = data_raw['trPL'].idxmax()

            # remove everything before the maximum
            data_raw = data_raw.iloc[max_idx:]
            # reset the index
            data_raw = data_raw.dropna() # remove rows with negative trPL values
            data_raw = data_raw[data_raw['trPL'] >= 0]# remove rows with negative trPL values
            data_raw['t'] = data_raw['t'] - data_raw['t'][max_idx]
            data_raw = data_raw[data_raw['t'] >= 0]# remove rows with negative trPL values
            data_raw = data_raw.reset_index(drop=True)

            # interpolate the data to have a logarithmically spaced time axis
            t_log = np.logspace(np.log10(data_raw['t'][1]), np.log10(data_raw['t'].max()), num=1000)
            t_log = np.insert(t_log, 0, 0)  # add 0 to the time array
            # interpolate the trPL values
            trPL_log = np.interp(t_log, data_raw['t'] - data_raw['t'].min(), data_raw['trPL'])

            power.append(float(file.split('_')[-1].replace('uW.dat', '')))
            if idx == 0:
                data2fit = {'t': t_log - t_log.min(), 'trPL': trPL_log, 'G_frac': power[-1] / pmax * np.ones_like(t_log)}
            else:
                data2fit['t'] = np.concatenate((data2fit['t'], t_log - t_log.min()))
                data2fit['trPL'] = np.concatenate((data2fit['trPL'], trPL_log))
                data2fit['G_frac'] = np.concatenate((data2fit['G_frac'], power[-1] / pmax * np.ones_like(t_log)))

        data2fit= pd.DataFrame(data2fit)

        time = data2fit['t'].values # time in seconds
        X = np.asarray(data2fit[['t', 'G_frac']])
        y = np.asarray(data2fit['trPL'])
        fpu = 10e3 # Frequency of the pump laser in Hz
        N0 = 1.39e+20
        background = 0e28 # Background illumination 

        # Define the Agent and the target metric/loss function
        metric = 'nrmse'
        loss = 'linear' # 'nrmse' or 'mse' or 'soft_l1' or 'linear'
        pump_args = {'N0': N0, 'fpu': fpu , 'background' : background, }
        exp_format = 'trPL' # experiment format
        model = partial(DBTD_multi_trap, method='LSODA', dimensionless=False, timeout=30, timeout_solve=30)
        RateEq = RateEqAgent(params, [X], [y], model = model, pump_model = initial_carrier_density, pump_args = pump_args, fixed_model_args = {}, metric = metric, loss = loss,minimize=True,exp_format=exp_format,detection_limit=0e-5,  transforms =['normalize','log'], do_G_frac_transform=True)

        model_gen_kwargs_list = None
        # Here we add some constraints to the parameters to help the optimizer
        parameter_constraints = [f' -N_t_bulk_1 - 0.5 * C_n_1 - 0.5 * C_p_1  - N_t_bulk_2 - 0.5 * C_n_2 - 0.5 * C_p_2<= -5']

        model_kwargs_list = [{},{"torch_device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),'botorch_acqf_class':qLogNoisyExpectedImprovement,'transforms':[RemoveFixed, Log,UnitX, StandardizeY],'surrogate_spec':SurrogateSpec(model_configs=[ModelConfig(botorch_model_class=SingleTaskGP,covar_module_class=ScaleKernel, covar_module_options={'base_kernel':MaternKernel(nu=2.5, ard_num_dims=num_free_params)})])}]

        optimizer = axBOtorchOptimizer(params = params, agents = RateEq, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,2], batch_size = [8,4], ax_client = None,  max_parallelism = 100, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = model_gen_kwargs_list, name = 'ax_opti',parameter_constraints = parameter_constraints,parallel_agents= True)

        optimizer.optimize_turbo(force_continue=False,kwargs_turbo_state={'failure_tolerance':4}) # run the optimization with turbo

        assert True
    except Exception as e:
        assert False, "Error occurred during multi-trap rate equation fitting: {}".format(e)

def test_diff_rec():

    try:
        # Define the parameters to be fitted
        # Note: in general for log spaced value it is better to use the foce_log option when doing the Bayesian inference
        params = []

        k_direct = FitParam(name = 'k_direct', value = 7.5e-17, bounds = [1e-18,1e-15], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{direct}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(k_direct)

        k_deep = FitParam(name = 'k_deep', value = 1.6e4, bounds = [1e4,1e6], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{deep}}$', unit='s$^{-1}$', axis_type = 'log', force_log=True)
        params.append(k_deep)

        k_c = FitParam(name = 'k_c', value = 1.3e6, bounds = [1e4,1e8], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{c}}$', unit='s$^{-1}$', axis_type = 'log', force_log=True)
        params.append(k_c)

        k_e = FitParam(name = 'k_e', value = 7.5e5, bounds = [1e4,1e8], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{e}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(k_e)

        mu = FitParam(name = 'mu', value = 4e-1*1e-4, bounds = [1e-6,1e-2], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$\mu$', unit='m$^{2}$ V$^{-1}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(mu)

        Sfront = FitParam(name = 'S_front', value = 7*1e-2, bounds = [1e-4,1e-1], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$S_{\text{front}}$', unit='m s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(Sfront)

        L = FitParam(name = 'L', value = 600e-9, bounds = [400e-9,1e-6], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$L$', unit='m', axis_type = 'linear',force_log=True)
        params.append(L)

        Sback = FitParam(name = 'S_back', value = 4*1e-2, bounds = [1e-4,1e-1], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$S_{\text{back}}$', unit='m s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(Sback)

        I_factor_PL = FitParam(name = 'I_factor_PL', value = 1e-28, bounds = [1e-29,1e-27], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$I_{\text{PL}}$', unit='-', axis_type = 'log', force_log=True)
        params.append(I_factor_PL)

        N_A = FitParam(name = 'N_A', value = 2.1e21, bounds = [1e20,1e23], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$N_A$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
        params.append(N_A)

        alpha = FitParam(name = 'alpha', value = 3e8, bounds = [1e6,1e8], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$\alpha$', unit='m$^{-1}$', axis_type = 'log',)
        params.append(alpha)

        # original values
        params_orig = copy.deepcopy(params)
        num_free_params = 0
        dum_dic = {}
        for i in range(len(params)):
            if params[i].force_log:
                dum_dic[params[i].name] = np.log10(params[i].value)
            else:
                dum_dic[params[i].name] = params[i].value/params[i].fscale
        # we need this just to run the model to generate some fake data

            if params[i].type != 'fixed':
                num_free_params += 1

        # Define the path to the data
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        path2data  = os.path.join(parent_dir,'Data','perovskite_trPL')
        data_raw = pd.read_csv(os.path.join(path2data,'Seo_FAPI1_Glass_0.dat'), sep=r'\s+',names=['t',"trPL"], skiprows=1)
        # convert time to seconds
        data_raw['t'] = data_raw['t'] * 1e-9 # convert to seconds
        max_idx = data_raw['trPL'].idxmax()
        # remove everything before the maximum
        data_raw = data_raw.iloc[max_idx-1:]
        # reset the index
        data_raw = data_raw.dropna() # remove rows with negative trPL values
        data_raw = data_raw[data_raw['trPL'] >= 0]# remove rows with negative trPL values
        data_raw = data_raw.reset_index(drop=True)

        # interpolate the data to have a logarithmically spaced time axis
        t_log = np.logspace(np.log10(2e-9), np.log10(data_raw['t'].max()), num=1000)
        t_log = np.insert(t_log, 0, 0)  # add 0 to the time array
        # interpolate the trPL values
        trPL_log = np.interp(t_log, data_raw['t'] - data_raw['t'].min(), data_raw['trPL'])
        data2fit = pd.DataFrame({'t': t_log - t_log.min(), 'trPL': trPL_log})

        # Plot the data to be fitted and the initial guess
        tim = data2fit['t'].values # time in seconds
        X = tim
        y = data2fit['trPL'] 

        fpu = 50e3 # Frequency of the pump laser in Hz
        Fluence = 4.8e15 # Fluence in m-2
        z_array = np.linspace(0, L.value, 100) # z-axis in meters
        generation = np.exp(-alpha.value * z_array)
        generation_sum = np.trapezoid(generation, z_array)
        n_0z = Fluence / generation_sum * generation
        N0 = np.mean(n_0z) # mean initial carrier density in m-3
        background = 0e28 # Background illumination 


        # Define the Agent and the target metric/loss function
        metric = 'nrmse'
        loss = 'linear' # 'nrmse' or 'mse' or 'soft_l1' or 'linear'
        pump_args = {'N0': N0, 'fpu': fpu , 'background' : background, }
        exp_format = 'trPL' # experiment format
        RateEq = RateEqAgent(params, [X], [y], model = DBTD_model, pump_model = initial_carrier_density, pump_args = pump_args, fixed_model_args = {}, metric = metric, loss = loss,minimize=True,exp_format=exp_format,detection_limit=0e-5,  transforms='log')

        model_gen_kwargs_list = None
        parameter_constraints = None

        model_kwargs_list = [{},{"torch_device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),'botorch_acqf_class':qLogNoisyExpectedImprovement,'transforms':[RemoveFixed, Log,UnitX, StandardizeY],'surrogate_spec':SurrogateSpec(model_configs=[ModelConfig(botorch_model_class=SingleTaskGP,covar_module_class=ScaleKernel, covar_module_options={'base_kernel':MaternKernel(nu=2.5, ard_num_dims=num_free_params)})])}]

        optimizer = axBOtorchOptimizer(params = params, agents = RateEq, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,2], batch_size = [8,2], ax_client = None,  max_parallelism = 100, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = model_gen_kwargs_list, name = 'ax_opti',parameter_constraints = parameter_constraints,)

        optimizer.optimize_turbo(force_continue=True,kwargs_turbo_state={'failure_tolerance':10}) # run the optimization with turbo
    
        assert True
    except Exception as e:
        assert False, "Error occurred during diffusion-recombination fitting: {}".format(e)