"""axBOtorchOptimizer module. This module contains the axBOtorchOptimizer class. The class is used to run the bayesian optimization process using the Ax library."""
######### Package Imports #########################################################################

import numpy as np
from joblib import Parallel, delayed
from functools import partial
from optimpv import *
from optimpv.axBOtorch.axUtils import *
from optimpv.axBOtorch.axUtils import *
from optimpv.axBOtorch.axSchedulerUtils import *
import ax, os, shutil
from ax import *
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax import Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.scheduler import Scheduler, SchedulerOptions, TrialType
from collections import defaultdict
from torch.multiprocessing import Pool, set_start_method
try: # needed for multiprocessing when using pytorch
    set_start_method('spawn')
except RuntimeError:
    pass

from logging import Logger
from ax.utils.common.logger import get_logger, _round_floats_for_logging

logger: Logger = get_logger(__name__)
ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)

######### Optimizer Definition #######################################################################
class axBOtorchOptimizer():
    """Initialize the axBOtorchOptimizer class. The class is used to run the optimization process using the Ax library. 

    Parameters
    ----------
    params : list of Fitparam() objects, optional
        List of Fitparam() objects, by default None
    agents : list of Agent() objects, optional
        List of Agent() objects see optimpv/general/BaseAgent.py for a base class definition, by default None
    models : list, optional
        list of models to use for the optimization process, by default ['SOBOL','BOTORCH_MODULAR']
    n_batches : list, optional
        list of the number of batches for each model, by default [1,10]
    batch_size : list, optional
        list of the batch sizes for each model, by default [10,2]
    ax_client : AxClient, optional
        AxClient object, by default None
    max_parallelism : int, optional
        maximum number of parallel processes to run, by default 10
    model_kwargs_list : dict, optional
        dictionary of model kwargs for each model, by default None
    model_gen_kwargs_list : dict, optional
        dictionary of model generation kwargs for each model, by default None
    name : str, optional
        name of the optimization process, by default 'ax_opti'

    Raises
    ------
    ValueError
        raised if the number of batches and the number of models are not the same
    ValueError
        raised if the model is not a string or a Models enum
    ValueError
        raised if the model_kwargs_list and models do not have the same length
    ValueError
        raised if the model_gen_kwargs_list and models do not have the same length
    """ 
    def __init__(self, params = None, agents = None, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,10], batch_size = [10,2], ax_client = None,  max_parallelism = 10,model_kwargs_list = None, model_gen_kwargs_list = None, name = 'ax_opti', **kwargs):
               
        self.params = params
        if not isinstance(agents, list):
            agents = [agents]
        self.agents = agents
        self.models = models
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.all_metrics = None
        self.ax_client = ax_client
        self.max_parallelism = max_parallelism
        if max_parallelism == -1:
            self.max_parallelism = os.cpu_count()-1
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
        """ Create a generation strategy for the optimization process using the models and the number of batches and batch sizes. See ax documentation for more details: https://ax.dev/tutorials/generation_strategy.html

        Returns
        -------
        GenerationStrategy
            The generation strategy for the optimization process

        Raises
        ------
        ValueError
            If the model is not a string or a Models enum
        """        

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
                model_gen_kwargs= self.model_gen_kwargs_list[i],
            ))

        gs = GenerationStrategy(steps=steps, )

        return gs

    def create_generation_strategy_batch(self):
        """ Create a generation strategy for the optimization process using the models and the number of batches and batch sizes. See ax documentation for more details: https://ax.dev/tutorials/generation_strategy.html

        Returns
        -------
        GenerationStrategy
            The generation strategy for the optimization process

        Raises
        ------
        ValueError
            If the model is not a string or a Models enum
        """        

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
                num_trials=self.n_batches[i],#*self.batch_size[i],
                max_parallelism=min(self.max_parallelism,self.batch_size[i]),
                model_kwargs= self.model_kwargs_list[i],
                model_gen_kwargs= self.model_gen_kwargs_list[i],
            ))

        gs = GenerationStrategy(steps=steps, )

        return gs
    
    def create_objectives(self):
        """ Create the objectives for the optimization process. The objectives are the metrics of the agents. The objectives are created using the metric, minimize and threshold attributes of the agents. If the agent has an exp_format attribute, it is used to create the objectives.

        Returns
        -------
        dict
            A dictionary of the objectives for the optimization process
        """        

        append_metrics = False
        if self.all_metrics is None:
            self.all_metrics = []
            append_metrics = True
        objectives = {}
        for agent in self.agents:
            for i in range(len(agent.metric)):
                # if exp_format is an attribute of the agent, use it
                if hasattr(agent,'exp_format'):
                    objectives[agent.name+'_'+agent.exp_format[i]+'_'+agent.metric[i]] = ObjectiveProperties(minimize=agent.minimize[i], threshold=agent.threshold[i])
                    if append_metrics:
                        self.all_metrics.append(agent.name+'_'+agent.exp_format[i]+'_'+agent.metric[i])
                else:
                    objectives[agent.name+'_'+agent.metric[i]] = ObjectiveProperties(minimize=agent.minimize[i], threshold=agent.threshold[i])
                    if append_metrics:
                        self.all_metrics.append(agent.name+'_'+agent.metric[i])

        return objectives
    
    def evaluate(self,args):
        """ Evaluate the agent on a parameter point

        Parameters
        ----------
        args : tuple
            Tuple containing the index of the agent, the agent, the index of the parameter point and the parameter point

        Returns
        -------
        tuple
            Tuple containing the index of the parameter point and the results of the agent on the parameter point
        """        
        idx, agent, p_idx, p = args
        res = agent.run_Ax(p)
        return p_idx, res
    
    def optimize(self,batch=False):
        """ Run the optimization process using the agents and the parameters. The optimization process uses the Ax library. The optimization process runs the agents in parallel if the parallel_agents attribute is True. The optimization process runs using the parameters, agents, models, n_batches, batch_size, max_parallelism, model_kwargs_list, model_gen_kwargs_list, name and kwargs attributes of the class. The optimization process runs using the create_generation_strategy and create_objectives methods of the class. The optimization process runs using the run_Ax method of the agents.

        Parameters
        ----------
        batch : bool, optional
            If True, run the optimization process in batch mode. The default is False.

        Raises
        ------
        ValueError
            If the number of batches and the number of models are not the same

        """  
        if batch:
            self.optimize_batch()
        else:
            self.optimize_sequential()

    def optimize_sequential(self):
        """ Run the optimization process using the agents and the parameters. The optimization process uses the Ax library. The optimization process runs the agents in parallel if the parallel_agents attribute is True. The optimization process runs using the parameters, agents, models, n_batches, batch_size, max_parallelism, model_kwargs_list, model_gen_kwargs_list, name and kwargs attributes of the class. The optimization process runs using the create_generation_strategy and create_objectives methods of the class. The optimization process runs using the run_Ax method of the agents.

        Raises
        ------
        ValueError
            If the number of batches and the number of models are not the same

        """        

        # from kwargs
        enforce_sequential_optimization = self.kwargs.get('enforce_sequential_optimization',False)
        global_max_parallelism = self.kwargs.get('global_max_parallelism',-1)
        verbose_logging = self.kwargs.get('verbose_logging',True)
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
            self.ax_client = AxClient(generation_strategy=gs, enforce_sequential_optimization=enforce_sequential_optimization, verbose_logging=verbose_logging,global_stopping_strategy=global_stopping_strategy)
        
        
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
                    dum_res = Parallel(n_jobs=min(curr_batch_size*len(self.agents),self.max_parallelism))(delayed(agent.run_Ax)(p) for p in parameters.values())
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
                with Pool(processes=min(len(agent_param_list),self.max_parallelism)) as pool:
                    parallel_results = pool.map(self.evaluate, agent_param_list)

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


    # def call_logger(self,curr_batch_size,count):
        # logger.info(f'Finished batch "{round_floats_for_logging(count)}" and starting batch "{round_floats_for_logging(count+1)}" with "{round_floats_for_logging(curr_batch_size)}" trials')
     
    def optimize_batch(self):
        """ Run the optimization process using the agents and the parameters. The optimization process uses the Ax library. The optimization process runs the agents in parallel if the parallel_agents attribute is True. The optimization process runs using the parameters, agents, models, n_batches, batch_size, max_parallelism, model_kwargs_list, model_gen_kwargs_list, name and kwargs attributes of the class. The optimization process runs using the create_generation_strategy and create_objectives methods of the class. The optimization process runs using the run_Ax method of the agents.

        Raises
        ------
        ValueError
            If the number of batches and the number of models are not the same

        """

        # from kwargs
        enforce_sequential_optimization = self.kwargs.get('enforce_sequential_optimization',False)
        global_max_parallelism = self.kwargs.get('global_max_parallelism',-1)
        verbose_logging = self.kwargs.get('verbose_logging',True)
        scheduler_logging_level = self.kwargs.get('scheduler_logging_level',0)
        global_stopping_strategy = self.kwargs.get('global_stopping_strategy',None)
        outcome_constraints = self.kwargs.get('outcome_constraints',None)
        parameter_constraints = self.kwargs.get('parameter_constraints',None)
        parallel_agents = self.kwargs.get('parallel_agents',True)
        max_number_cores = self.kwargs.get('max_number_cores',-1)
        init_seconds_between_polls = self.kwargs.get('init_seconds_between_polls',0.1)
        logging_level = self.kwargs.get('logging_level',20)
        keep_tmp_dir = self.kwargs.get('keep_tmp_dir',False)

        if max_number_cores == -1:
            max_number_cores = os.cpu_count()-1
        tmp_dir = self.kwargs.get('tmp_dir',None)
        tmp_dir = os.path.join(os.getcwd(),'.tmp_dir') if tmp_dir is None else tmp_dir

        # create parameters space from params
        parameters_space = ConvertParamsAx(self.params)

        # create generation strategy
        gs = self.create_generation_strategy_batch()
        # gs.use_batch_trials = True

        # create ax client
        if self.ax_client is None:
            self.ax_client = AxClient(generation_strategy=gs, enforce_sequential_optimization=enforce_sequential_optimization, verbose_logging=verbose_logging,global_stopping_strategy=global_stopping_strategy)
        
        _obj = self.create_objectives()

        is_multi_obj = False
        if len(_obj.keys()) > 1:
            is_multi_obj = True

        q = Pool(max_number_cores)
        if not is_multi_obj:
            # obj = Objective(metric=BraninForMockJobMetric(name=list(_obj.keys())[0]+'_', agents = self.agents, pool = q, tmp_dir = tmp_dir, parallel_agents = parallel_agents), minimize=True)
            obj = Objective(metric=BraninForMockJobMetric(name=list(_obj.keys())[0], agents = self.agents, pool = q, tmp_dir = tmp_dir, parallel_agents = parallel_agents), minimize=_obj[list(_obj.keys())[0]].minimize)
        else:
            objectives_list = []
            objectives_thresholds = []

            for key in _obj.keys():
                lower_is_better = _obj[key].minimize
                metric = BraninForMockJobMetric(name=key, agents = self.agents, pool = q, tmp_dir = tmp_dir, parallel_agents = parallel_agents,lower_is_better=lower_is_better)
                objectives_list.append(Objective(metric=metric, minimize=lower_is_better))
                objectives_thresholds.append(ObjectiveThreshold(metric=metric, bound=_obj[key].threshold,relative=False))
            obj = MultiObjective(objectives=objectives_list, objective_thresholds=objectives_thresholds)
            # raise ValueError('The objective must be a single metric')

        # create experiment
        self.ax_client.create_experiment(
            name=self.name,
            parameters=parameters_space,
            # objectives=self.create_objectives(),
            outcome_constraints=outcome_constraints,
            parameter_constraints=parameter_constraints,
            
        )
        # threshold=
        if not is_multi_obj:
            self.ax_client.experiment.optimization_config=OptimizationConfig(objective=obj)
        else:
            self.ax_client.experiment.optimization_config=MultiObjectiveOptimizationConfig(objective=obj)
            # self.ax_client.experiment.optimization_config.objective_thresholds = objectives_thresholds
        
        # create runner
        runner = MockJobRunner(agents = self.agents, pool = q, tmp_dir = tmp_dir, parallel_agents = parallel_agents)
        self.ax_client.experiment.runner = runner
        # run optimization
        n = 0
        total_trials = sum(np.asarray(self.n_batches)*np.asarray(self.batch_size))
        n_step_points = np.cumsum(np.asarray(self.n_batches)*np.asarray(self.batch_size))
        if verbose_logging:
            logger.info('Starting optimization with %d batches and a total of %d trials',sum(np.asarray(self.n_batches)),total_trials)
        count = 1
        while n < total_trials:
            # check the current batch size
            if n == 0:
                old_batch_size = self.batch_size[np.argmax(n_step_points>n)]
                # logger.info('Starting first batch with %d trials',old_batch_size)
            else:
                old_batch_size = curr_batch_size
                # logger.info('Finished batch %d and starting batch %d with %d trials',count-1,count,curr_batch_size)
            curr_batch_size = self.batch_size[np.argmax(n_step_points>n)]

            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                if n == 0:
                    logger.info(f'Starting batch {round_floats_for_logging(count)} with {round_floats_for_logging(curr_batch_size)} trials')
                else:
                    logger.info(f'Finished batch {round_floats_for_logging(count)} and starting batch {round_floats_for_logging(count+1)} with {round_floats_for_logging(curr_batch_size)} trials')

            curr_batch_size = self.batch_size[np.argmax(n_step_points>n)]

            if old_batch_size != curr_batch_size or n == 0: # if the batch size changes, create a new scheduler
                # Create a new scheduler for each batch with the current batch size
                scheduler = Scheduler(
                    experiment=self.ax_client.experiment,
                    generation_strategy=self.ax_client.generation_strategy,
                    options=SchedulerOptions(run_trials_in_batches=True,init_seconds_between_polls=init_seconds_between_polls,trial_type=TrialType.BATCH_TRIAL,batch_size=curr_batch_size,logging_level=scheduler_logging_level,global_stopping_strategy=global_stopping_strategy),
                )

            n += curr_batch_size
            if n > total_trials:
                curr_batch_size = curr_batch_size - (n-total_trials)
            
            scheduler.run_n_trials(max_trials=1)
            count += 1
            
            
        q.close()
        q.join()
        if verbose_logging:
            logging_level = 20
            logger.setLevel(logging_level)
            logger.info('Finished optimization with %d number of batches and a total of %d trials',len(self.n_batches),total_trials)

        # clean up the tmp_dir
        if not keep_tmp_dir:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    pass