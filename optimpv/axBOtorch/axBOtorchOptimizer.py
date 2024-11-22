
import numpy as np
from joblib import Parallel, delayed
from functools import partial,reduce
from optimpv import *
from optimpv.axBOtorch.axUtils import *
import ax
from ax import *
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax import Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.modelbridge.transforms.unit_x import UnitX
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.modelbridge.transforms.log import Log
# from joblib import Parallel, delayed
from collections import defaultdict
from torch.multiprocessing import Pool, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
# from multiprocessing import Pool

class axBOtorchOptimizer():
    def __init__(self, params = None, agents = None, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,10], batch_size = [10,2], metrics = None, minimize_list = None, thresholds = None, ax_client = None,  max_parallelism = 10,model_kwargs_list = None, model_gen_kwargs_list = None, name = 'ax_opti', **kwargs):
        self.params = params
        if not isinstance(agents, list):
            agents = [agents]
        self.agents = agents
        self.models = models
        self.n_batches = n_batches
        self.batch_size = batch_size

        if metrics is None:
            metrics = [agent.metric for agent in agents]
        elif isinstance(metrics,str):
            metrics = [metrics]*len(agents)
        else:
            if len(metrics) != len(agents):
                raise ValueError('Metrics must have the same length as agents or be a string')
            for i, metric in enumerate(metrics):
                if isinstance(metric,str):
                    metrics[i] = [metric]
                elif not isinstance(metric,list):
                    raise ValueError('Metrics must be a string or a list of strings')
        self.metrics = metrics
        # match the shape of minimize_list with metrics
        if minimize_list is None:
            if isinstance(metrics[0],list):
                for metric in metrics:
                    if type(metric) == str:
                        minimize_list.append([True])
                    else:
                        minimize_list.append([True]*len(metric))   
            else:
                minimize_list = [True]*len(metrics)
        elif isinstance(minimize_list,bool):
            dum_bool = minimize_list
            if isinstance(metrics[0],list):
                minimize_list = []
                for metric in metrics:
                    if type(metric) == str:
                        minimize_list.append([dum_bool])
                    else:
                        minimize_list.append([dum_bool]*len(metric))
            else:
                minimize_list = [dum_bool]*len(metrics)
        else:
            # check that all elements from the list are the same size than in metrics
            for i, metric in enumerate(metrics):
                if len(minimize_list[i]) != len(metric):
                    raise ValueError('Minimize_list must have the same shape as metrics')
                
        self.minimize_list = minimize_list
        # do the same for thresholds than for minimize_list
        if thresholds is None:
            if isinstance(metrics[0],list):
                thresholds = []
                for metric in metrics:
                    if type(metric) == str:
                        thresholds.append([None])
                    else:
                        thresholds.append([None]*len(metric))
            else:
                thresholds = [None]*len(metrics)
        elif isinstance(thresholds,(int,float)):
            dum_thresh = thresholds
            if isinstance(metrics[0],list):
                thresholds = []
                for metric in metrics:
                    if type(metric) == str:
                        thresholds.append([dum_thresh])
                    else:
                        thresholds.append([dum_thresh]*len(metric))
            else:
                thresholds = [dum_thresh]*len(metrics)
        else:
            # check that all elements from the list are the same size than in metrics
            for i, metric in enumerate(metrics):
                if len(thresholds[i]) != len(metric):
                    raise ValueError('Thresholds must have the same shape as metrics')
        self.thresholds = thresholds
        self.ax_client = ax_client
        self.max_parallelism = max_parallelism
        if model_kwargs_list is None:
            model_kwargs_list = [{}]*len(models)
        elif isinstance(model_kwargs_list,dict):
            model_kwargs_list = [model_kwargs_list]*len(models)
        elif len(model_kwargs_list) != len(models):
            raise ValueError('model_kwargs_list must have the same length as models')
        self.model_kwargs_list = model_kwargs_list
        if model_gen_kwargs_list is None:
            model_gen_kwargs_list = [{}]*len(models)
        elif isinstance(model_gen_kwargs_list,dict):
            model_gen_kwargs_list = [model_gen_kwargs_list]*len(models) 
        elif len(model_gen_kwargs_list) != len(models):
            raise ValueError('model_gen_kwargs_list must have the same length as models')
        self.model_gen_kwargs_list = model_gen_kwargs_list
        self.name = name
        self.kwargs = kwargs

        if len(n_batches) != len(models):
            raise ValueError('n_batches and models must have the same length')
        if type(batch_size) == int:
            self.batch_size = [batch_size]*len(models)
        if len(batch_size) != len(models):
            raise ValueError('batch_size and models must have the same length')


    def create_generation_strategy(self):

    
        steps = []
        for i, model in enumerate(self.models):
            if type(model) == str:
                model = Models[model]
            elif isinstance(model, Models):
                model = model
            else:
                raise ValueError('Model must be a string or a Models enum')
            
            steps.append(GenerationStep(
                model=model,
                num_trials=self.n_batches[i]*self.batch_size[i],
                max_parallelism=min(self.max_parallelism,self.batch_size[i]),
                model_kwargs= self.model_kwargs_list[i],
                model_gen_kwargs= self.model_gen_kwargs_list[i]
            ))

        gs = GenerationStrategy(steps=steps, )

        return gs

    def create_objectives(self):

        if self.metrics is None:
            raise ValueError('No metrics defined')
        
        all_metrics, all_minimize, all_thresholds = [],[],[]
        # reduce the list of metrics to a single list
        
        all_metrics = reduce(lambda x,y: x+y, self.metrics)
        if not isinstance(all_metrics,list):
            all_metrics = [all_metrics]

        # check for duplicates
        if len(all_metrics) != len(set(all_metrics)):
            self.add_agent_index = True
            all_metrics = []
            # create a list of all metrics
            for i, metric in enumerate(self.metrics):
                if isinstance(metric,list):
                    for j, sub_metric in enumerate(metric):
                        all_metrics.append(str(i)+'_'+sub_metric)
                        all_minimize.append(self.minimize_list[i][j])
                        all_thresholds.append(self.thresholds[i][j])
                else:
                    all_metrics.append(str(i)+'_'+metric)
                    all_minimize.append(self.minimize_list[i])
                    all_thresholds.append(self.thresholds[i])
        else:
            self.add_agent_index = False
            all_minimize = reduce(lambda x,y: x+y, self.minimize_list)
            all_thresholds = reduce(lambda x,y: x+y, self.thresholds)

        self.all_metrics = all_metrics
        if not isinstance(all_minimize,list):
            all_minimize = [all_minimize]
        if not isinstance(all_thresholds,list):
            all_thresholds = [all_thresholds]

        objectives = {}
        for i, metric in enumerate(all_metrics):
            objectives[metric] = ObjectiveProperties(minimize=all_minimize[i], threshold=all_thresholds[i])

        return objectives

    def evaluate(self, parameters, agents ):
        
        dic_res = {}
        for i, agent in enumerate(agents):
            # update dic res with agent results
            res_dic = agent.run_Ax(parameters)
            if len(self.all_metrics) != len(set(self.all_metrics)):
                for key in res_dic.keys():
                    dic_res[str(i)+'_'+key] = res_dic[key]
            else:
                dic_res = res_dic

        return dic_res
    
    def run_agent_param(self,args):
        idx, agent, p_idx, p = args
        res = agent.run_Ax(p)
        if self.add_agent_index:
            res = {f"{idx}_{k}": v for k, v in res.items()}
        return p_idx, res
    
    def optimize(self):

        # from kwargs
        enforce_sequential_optimization = self.kwargs.get('enforce_sequential_optimization',False)
        global_max_parallelism = self.kwargs.get('global_max_parallelism',-1)
        verbose_logging = self.kwargs.get('verbose_logging',False)
        global_stopping_strategy = self.kwargs.get('global_stopping_strategy',None)
        outcome_constraints = self.kwargs.get('outcome_constraints',None)
        parameter_constraints = self.kwargs.get('parameter_constraints',None)
        parallel_agents = self.kwargs.get('parallel_agents',True)

        # create parameters space from params
        parameters_space = ConvertParamsAx(self.params)

        # create generation strategy
        gs = self.create_generation_strategy()

        # create ax client
        if self.ax_client is None:
            self.ax_client = AxClient(generation_strategy=gs, enforce_sequential_optimization=enforce_sequential_optimization)
        
        
        # create experiment
        self.ax_client.create_experiment(
            name=self.name,
            parameters=parameters_space,
            objectives=self.create_objectives(),
            outcome_constraints=outcome_constraints,
            parameter_constraints=parameter_constraints,
        )


        # run optimization
        n = 0
        total_trials = sum(np.asarray(self.n_batches)*np.asarray(self.batch_size))
        n_step_points = np.cumsum(np.asarray(self.n_batches)*np.asarray(self.batch_size))

        while n < total_trials:
            # check the current batch size
            curr_batch_size = self.batch_size[np.argmax(n_step_points>n)]
            n += curr_batch_size
            if n > total_trials:
                curr_batch_size = curr_batch_size - (n-total_trials)

            parameters, trial_index = self.ax_client.get_next_trials(curr_batch_size)

            if not parallel_agents:
                results = []
                for idx, agent in enumerate(self.agents):
                    dum_res = Parallel(n_jobs=curr_batch_size)(delayed(agent.run_Ax)(p) for p in parameters.values())
                    if self.add_agent_index:
                        res_lst = []
                        for res in dum_res:
                            new_dict = {}
                            for key in res.keys():
                                new_dict[str(idx)+'_'+key] = res[key]
                            res_lst.append(new_dict)
                        results.append(res_lst)
                    else:
                        results.append(dum_res)
                
                main_results = []
                # merge the n agents results
                for i in range(len(results[0])):
                    main_results.append({})
                    for j in range(len(results)):
                        main_results[-1].update(results[j][i])
            else:
                agent_param_list =[]
                for p_idx, p in enumerate(parameters.values()):
                    for idx, agent in enumerate(self.agents):
                        agent_param_list.append((idx, agent, p_idx, p))

                # Run all combinations in parallel using multiprocessing
                with Pool(processes=curr_batch_size) as pool:
                    parallel_results = pool.map(self.run_agent_param, agent_param_list)

                # Collect and merge results
                results_dict = defaultdict(dict)
                for p_idx, res in parallel_results:
                    results_dict[p_idx].update(res)

                # Convert to main_results list
                main_results = [results_dict[i] for i in sorted(results_dict)]

            for trial_index, raw_data in zip(parameters.keys(), main_results):
                got_nan = False
                for key in raw_data.keys():
                    if np.isnan(raw_data[key]):
                        got_nan = True
                        break
                if not got_nan:
                    self.ax_client.complete_trial(trial_index, raw_data=raw_data)
                else:
                    self.ax_client.log_trial_failure(trial_index)

