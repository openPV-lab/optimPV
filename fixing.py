######### Package Imports #########################################################################
import os, warnings, copy, torch, ax, uuid,json,time
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from joblib import Parallel, delayed
from numpy.random import default_rng
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.modelbridge.transforms.unit_x import UnitX
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.modelbridge.transforms.log import Log
from ax.core.base_trial import TrialStatus as T
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.plot.slice import plot_slice

from optimpv import *
from optimpv.axBOtorch.axBOtorchOptimizer import axBOtorchOptimizer
from optimpv.RateEqfits.RateEqAgent import RateEqAgent
from optimpv.RateEqfits.RateEqModel import *
from optimpv.RateEqfits.Pumps import *

# init_notebook_plotting()
warnings.filterwarnings('ignore') 


from random import randint
# from time import time
from typing import Any, Dict, NamedTuple, Union

from ax.core.base_trial import TrialStatus

from torch.multiprocessing import Pool, set_start_method
try: # needed for multiprocessing when using pytorch
    set_start_method('spawn')
except RuntimeError:
    pass


# class MockJob(NamedTuple):
#     """Dummy class to represent a job scheduled on `MockJobQueue`."""

#     id: int
#     parameters: Dict[str, Union[str, float, int, bool]]

#     def run(self, job_id, parameters, agents = None, tmp_dir = None, parallel_agents = True):
        
#         if parallel_agents:
#             res_dic = {}
#             results = Parallel(n_jobs=len(agents))(delayed(agents[i].run_Ax)(parameters) for i in range(len(agents)))
#             for i in range(len(agents)):
#                 res_dic.update(results[i])
#         else:
#             res_dic = {}
#             for i in range(len(agents)):
#                 res_dic.update(agents[i].run_Ax(parameters))

#         # save the results in tmp folder with the job_id in json format
#         if tmp_dir is not None:
#             if not os.path.exists(tmp_dir):
#                 os.makedirs(tmp_dir)
#             with open(os.path.join(tmp_dir,str(job_id)+'.json'), 'w') as fp:
#                 json.dump(res_dic, fp)
#         print('job_id:',job_id,'parameters:',parameters,'res_dic:',res_dic)


from collections import defaultdict
from typing import Iterable, Set

from ax.core.base_trial import BaseTrial
from ax.core.runner import Runner
from ax.core.trial import Trial



from optimpv.axBOtorch.axUtils import ConvertParamsAx, CreateObjectiveFromAgent
from ax import *


# search_space = SearchSpace(parameters=ConvertParamsAx(params))
def search_spaceAx(search_space):
    parameters = []
    for param in search_space:
        if param['type'] == 'range':
            if param['value_type'] == 'int':
                parameters.append(RangeParameter(name=param['name'], parameter_type=ParameterType.INT, lower=param['bounds'][0], upper=param['bounds'][1]))
            else:
                parameters.append(RangeParameter(name=param['name'], parameter_type=ParameterType.FLOAT, lower=param['bounds'][0], upper=param['bounds'][1]))
        elif param['type'] == 'fixed':
            if param['value_type'] == 'int':
                parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.INT, value=param.value))
            elif param['value_type'] == 'str':
                parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.STRING, value=param.value))
            elif param['value_type'] == 'bool':
                parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.BOOL, value=param.value))
            else:
                parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.FLOAT, value=param.value))
        elif param['type'] == 'choice':
            parameters.append(ChoiceParameter(name=param.name, values=param.values))
        else:
            raise ValueError('The parameter type is not recognized')
    return SearchSpace(parameters=parameters)



import pandas as pd

from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.utils.common.result import Ok, Err



import numpy as np
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render, init_notebook_plotting


import pySIMsalabim as sim
from pySIMsalabim.experiments.JV_steady_state import *
from optimpv.DDfits.JVAgent import JVAgent

from optimpv.axBOtorch.axBOtorchOptimizer_runner import *
def main():
    ##############################################################################################
    # Define the parameters to be fitted
    params = []

    mun = FitParam(name = 'l2.mu_n', value = 3e-5, bounds = [1e-5,1e-3], values = None, start_value = None, log_scale = True, value_type = 'float', fscale = None, rescale = False, stepsize = None, display_name=r'$\mu_n$', unit='m$^2$ V$^{-1}$s$^{-1}$', axis_type = 'log', std = 0,encoding = None,force_log = False)
    params.append(mun)

    mup = FitParam(name = 'l2.mu_p', value = 8e-4, bounds = [1e-5,1e-3], values = None, start_value = None, log_scale = True, value_type = 'float', fscale = None, rescale = False, stepsize = None, display_name=r'$\mu_p$', unit=r'm$^2$ V$^{-1}$s$^{-1}$', axis_type = 'log', std = 0,encoding = None,force_log = False)
    params.append(mup)

    bulk_tr = FitParam(name = 'l2.N_t_bulk', value = 1e20, bounds = [1e19,1e21], values = None, start_value = None, log_scale = True, value_type = 'float', fscale = None, rescale = False, stepsize = None, display_name=r'$N_{T}$', unit=r'm$^{-3}$', axis_type = 'log', std = 0,encoding = None,force_log = False)
    params.append(bulk_tr)

    int_trap = FitParam(name = 'l1.N_t_int', value = 4e12, bounds = [1e11,1e13], values = None, start_value = None, log_scale = True, value_type = 'float', fscale = None, rescale = False, stepsize = None, display_name=r'$N_{T,int}^{ETL}$', unit='m$^{-2}$', axis_type = 'log', std = 0,encoding = None,force_log = False)
    params.append(int_trap)

    Nions = FitParam(name = 'l2.N_ions', value = 1e22, bounds = [1e20,5e22], type='range', values = None, start_value = None, log_scale = True, value_type = 'float', fscale = None, rescale = False, stepsize = None, display_name=r'$C_{ions}$', unit='m$^{-3}$', axis_type = 'log', std = 0,encoding = None,force_log = False)
    params.append(Nions)

    R_series = FitParam(name = 'R_series', value = 1e-4, bounds = [1e-5,1e-3], type='range', values = None, start_value = None, log_scale = True, value_type = 'float', fscale = None, rescale = False, stepsize = None, display_name=r'$R_{series}$', unit=r'$\Omega$ m$^2$', axis_type = 'log', std = 0,encoding = None,force_log = False)
    params.append(R_series)

    # original values
    params_orig = copy.deepcopy(params)

    

    session_path = os.path.join('/home/lecorre/Desktop/pySIMsalabim/', 'SIMsalabim','SimSS')
    simss_device_parameters = os.path.join(session_path, 'simulation_setup.txt')

    # Set the JV parameters
    Gfracs = [0.1,0.5,1.0] # Fractions of the generation rate to simulate
    # Gfracs = None
    UUID = str(uuid.uuid4())

    cmd_pars = []
    for param in params:
        if param.name != 'l2.C_np_bulk' and param.name != 'offset_l2_l1.E_c' and param.name != 'offset_l2_l3.E_v' and param.name != 'Egap_l1.E_v' and param.name != 'offset_W_L.E_c' and param.name != 'l2.N_ions':
            cmd_pars.append({'par':param.name, 'val':str(param.value)})
        elif param.name == 'offset_l2_l1.E_c':
            cmd_pars.append({'par':'l1.E_c', 'val':str(3.9-param.value)})
            vv = 3.9-param.value
        elif param.name == 'l2.N_ions':
            cmd_pars.append({'par':'l2.N_cation', 'val':str(param.value)})
            cmd_pars.append({'par':'l2.N_anion', 'val':str(param.value)})
        elif param.name == 'l2.C_np_bulk':
            cmd_pars.append({'par':'l2.C_n_bulk', 'val':str(param.value)})
            cmd_pars.append({'par':'l2.C_p_bulk', 'val':str(param.value)})

        elif param.name == 'offset_l2_l3.E_v':
            cmd_pars.append({'par':'l3.E_v', 'val':str(5.53-param.value)})
        
        elif param.name == 'Egap_l1.E_v':
            cmd_pars.append({'par':'l1.E_v', 'val': str(vv+param.value)})
        
        elif param.name == 'offset_W_L.E_c':
            cmd_pars.append({'par':'W_L', 'val':str(vv-param.value)})


    print(cmd_pars)

    # Run the JV simulation
    ret, mess = run_SS_JV(simss_device_parameters, session_path, JV_file_name = 'JV.dat', varFile= 'Var.dat',G_fracs = Gfracs, parallel = True, max_jobs = 3, UUID=UUID, cmd_pars=cmd_pars)

    # import random noise
    from numpy.random import default_rng
    # save data for fitting
    X,y = [],[]
    if Gfracs is None:
        data = pd.read_csv(os.path.join(session_path, 'JV_'+UUID+'.dat'), sep=r'\s+') # Load the data
        Vext = np.asarray(data['Vext'].values)
        Jext = np.asarray(data['Jext'].values)
        G = np.ones_like(Vext)
        rng = default_rng()#
        noise = rng.standard_normal(Jext.shape) * 0.01 * Jext
        Jext = Jext + noise
        X= Vext
        y = Jext

        plt.figure()
        plt.plot(X,y)
        plt.show()
    else:
        for Gfrac in Gfracs:
            data = pd.read_csv(os.path.join(session_path, 'JV_Gfrac_'+str(Gfrac)+'_'+UUID+'.dat'), sep=r'\s+') # Load the data
            Vext = np.asarray(data['Vext'].values)
            Jext = np.asarray(data['Jext'].values)
            G = np.ones_like(Vext)*Gfrac
            rng = default_rng()#
            noise = rng.standard_normal(Jext.shape) * 0.01 * Jext
            Jext = Jext + noise
            if len(X) == 0:
                X = np.vstack((Vext,G)).T
                y = Jext
            else:
                X = np.vstack((X,np.vstack((Vext,G)).T))
                y = np.hstack((y,Jext))

        print(X.shape)
        print(y.shape)


        plt.figure()
        for Gfrac in Gfracs:
            plt.plot(X[X[:,1]==Gfrac,0],y[X[:,1]==Gfrac],label='Gfrac = '+str(Gfrac))
        plt.legend()
        # plt.show()



    
    metric = 'mse'
    # metric = 'mse'
    # loss = 'log10'
    # loss = 'linear'
    loss = 'soft_l1'

    jv = JVAgent(params, X, y, session_path, simss_device_parameters,parallel = True, max_jobs = 3, metric = metric, loss = loss)

    tmp_dir = os.path.join(os.getcwd(),'tmp')
    # create a pool 
    # q = Pool(3)

    

    # MOCK_JOB_QUEUE_CLIENT = MockJobQueueClient(agents = [jv], pool = q, tmp_dir = tmp_dir, parallel_agents = True)
    name = "RateEqRunner"
    search_space = search_spaceAx(ConvertParamsAx(params))
    
    

    objective = CreateObjectiveFromAgent(jv)
    keys_ = list(objective.keys())

    # def get_mock_job_queue_client(MOCK_JOB_QUEUE_CLIENT) -> MockJobQueueClient:
    #     """Obtain the singleton job queue instance."""
    #     return MOCK_JOB_QUEUE_CLIENT
    
    # get_mock_job_queue_client = partial(get_mock_job_queue_client, MOCK_JOB_QUEUE_CLIENT)


    


    objective = Objective(metric=BraninForMockJobMetric(name=keys_[0],agents = [jv], pool = q, tmp_dir = tmp_dir, parallel_agents = True), minimize=True)          
    # objective = CreateObjectiveFromAgent(RateEq)
    runner = MockJobRunner(agents = [jv], pool = q, tmp_dir = tmp_dir, parallel_agents = True)
    experiment = Experiment(name=name, search_space=search_space, optimization_config=OptimizationConfig(objective=objective), runner=runner)

    from ax.modelbridge.dispatch_utils import choose_generation_strategy

    generation_strategy = choose_generation_strategy(
        search_space=experiment.search_space,
        # max_parallelism_cap=5,
        use_batch_trials=True,
        max_parallelism_override=5, 
    )

    from ax.service.scheduler import Scheduler, SchedulerOptions


    # scheduler = Scheduler(
    #     experiment=experiment,
    #     generation_strategy=generation_strategy,
    #     options=SchedulerOptions(),
    # )


    # print(experiment.runner)
    model_gen_kwargs_list =[{},{'n':4}]
    parameter_constraints = [f'l2.mu_p - l2.mu_n >= {0}']

    model_kwargs_list = [{},{'torch_device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),'torch_dtype': torch.double,'botorch_acqf_class':qLogNoisyExpectedImprovement,'transforms':[RemoveFixed, Log,UnitX, StandardizeY],},] #,'surrogate':Surrogate(SingleTaskGP,covar_module_class=RBFKernel,covar_module_options={'lengthscale_constraint':Interval(1e-6,100)})}]
                        
    optimizer = axBOtorchOptimizer_runner(params = params, agents = jv, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,4], batch_size = [4,4], ax_client = None,  max_parallelism = 100,model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = model_gen_kwargs_list, name = 'ax_opti',parameter_constraints = parameter_constraints)

    optimizer._optimize()

    # sobol = Models.SOBOL(search_space=experiment.search_space)
    # for _ in range(5):
    #     generator_run = sobol.gen(n=1)
    #     # Add generator run to a trial to make it part of the experiment and evaluate arm(s) in it
    #     trial = experiment.new_trial(generator_run=generator_run)
    #     trial.run()
    #     # print(trial.status)
    #     while trial.status != T.COMPLETED:
            
    #         status_dict = experiment.runner.poll_trial_status([trial])
            
    #         if status_dict[T.COMPLETED]:
    #             trial.mark_completed()
    #         # # print(status_dict)

    #         time.sleep(0.1)
    #     # trial.mark_completed()

    # data = experiment.fetch_data()
    # print(data.df)
    # for i in range(10):
    #     model = Models.BOTORCH_MODULAR(experiment=experiment, data=data)
    #     generator_run = model.gen(3)
    #     trial = experiment.new_batch_trial(generator_run=generator_run)
    #     trial.run()
    #     # print(dir(trial))
    #     # time.sleep(1)
    #     jobs_ids_in_trial = [arm.run_metadata['job_id'] for arm in trial.arms]
    #     # for arm in trial.arms:
    #     #     print(arm.name)
    #     #     print(arm.run_metadata)
    #     # print(trial.run_metadata)
    #     # for job_id in jobs_ids_in_trial:
    #     #     print(job_id)
    #     #     print(MOCK_JOB_QUEUE_CLIENT.get_job_status(job_id))

    #     all_completed = all(
    #         runner.MOCK_JOB_QUEUE_CLIENT.get_job_status(job_id) == T.COMPLETED
    #         for job_id in jobs_ids_in_trial
    #     )
    #     while not all_completed:
    #         all_completed = all(
    #             runner.MOCK_JOB_QUEUE_CLIENT.get_job_status(job_id) == T.COMPLETED
    #             for job_id in jobs_ids_in_trial
    #         )
    #         print(all_completed)
    #         print([runner.MOCK_JOB_QUEUE_CLIENT.get_job_status(job_id) == T.COMPLETED
    #             for job_id in jobs_ids_in_trial])
    #         time.sleep(0.1)

    #     trial.mark_completed()
    #     data = Data.from_multiple_data([data, trial.fetch_data()])

    #     new_value = trial.fetch_data().df["mean"].min()
    #     print(
    #         f"Iteration: {i}, Best in iteration {new_value:.3f}, Best so far: {data.df['mean'].min():.3f}"
    #     )
    #     # raise ValueError('stop')
    #     # batch_trial = experiment.trials[trial.index]

    #     # # Check if all arms in the batch trial are completed
    #     # all_completed = all(
    #     #             arm_name in batch_trial._arm_trial_assignment
    #     #             and batch_trial.status_per_arm[arm_name] == TrialStatus.COMPLETED
    #     #             for arm_name in batch_trial.arm_names
    #     #         )
    #     # print(all_completed)
    #     # print(dir(trial))
    #     # for t in trial.generator_runs:
    #     #     print(dir(t.arms))
    #     #     print(type(t.arms[0]))
    #     # while trial.status != T.COMPLETED:
    #     #     print(trial.status)
    #         # status_dict = experiment.runner.poll_trial_status([trial])
    #         # # print(status_dict)
    #         # if status_dict[T.COMPLETED]:
    #         #     trial.mark_completed()
    #         #     while True:
    #         #         try:
    #         #             data = Data.from_multiple_data([data, trial.fetch_data()])
                        
    #         #             new_value = trial.fetch_data().df["mean"].min()
    #         #             print(
    #         #                 f"Iteration: {i}, Best in iteration {new_value:.3f}, Best so far: {data.df['mean'].min():.3f}"
    #         #             )
    #         #             break
    #         #         except:
    #         #             time.sleep(0.1)
    #         # print(status_dict)

    #         # time.sleep(0.1)


    #     # data = Data.from_multiple_data([data, trial.fetch_data()])

    #     # new_value = trial.fetch_data().df["mean"].min()
    #     # print(
    #     #     f"Iteration: {i}, Best in iteration {new_value:.3f}, Best so far: {data.df['mean'].min():.3f}"
    #     # )
    # # scheduler.run_n_trials(max_trials=10)
    # # scheduler.run_n_trials(max_trials=10)
    # # scheduler.run_n_trials(max_trials=10)
    # # scheduler.run_n_trials(max_trials=10)

    # # init_notebook_plotting()


    # # def get_plot():
    # #     best_objectives = np.array(
    # #         [[trial.objective_mean for trial in scheduler.experiment.trials.values()]]
    # #     )
    # #     best_objective_plot = optimization_trace_single_method(
    # #         y=np.minimum.accumulate(best_objectives, axis=1),
    # #         title="Model performance vs. # of iterations",
    # #         ylabel="Y",
    # #     )
    # #     return best_objective_plot

    # # render(get_plot())
    # # plt.show()
    from ax.service.utils.report_utils import exp_to_df
    ax_client = optimizer.ax_client
    experiment = ax_client.experiment
    df = exp_to_df(experiment)
    print(df)
    
    plt.figure()
    plt.plot(np.minimum.accumulate(df['JV_JV_mse'].values))
    plt.show()

    # terminate the pool
    # q.terminate()
    # # q.close()
    # q.join()
    # q = None   # Remove references to the pool

    # print('done')


if __name__ == '__main__':
    main()