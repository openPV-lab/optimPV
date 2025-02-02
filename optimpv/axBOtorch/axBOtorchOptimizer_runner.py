
import os,sys,json,uuid,time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from functools import partial,reduce
from typing import Any, Dict, NamedTuple, Union, Iterable, Set
from optimpv import *
from optimpv.axBOtorch.axUtils import *
from optimpv.axBOtorch.axUtils import ConvertParamsAx, CreateObjectiveFromAgent
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
from ax.runners.synthetic import SyntheticRunner
from ax.core.base_trial import BaseTrial
from ax.core.base_trial import TrialStatus
from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.core.data import Data
from ax.utils.common.result import Ok, Err
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.service.scheduler import Scheduler, SchedulerOptions, TrialType
from collections import defaultdict
from torch.multiprocessing import Pool, set_start_method
try: # needed for multiprocessing when using pytorch
    set_start_method('spawn')
except RuntimeError:
    pass


# def search_spaceAx(search_space):
#     parameters = []
#     for param in search_space:
#         if param['type'] == 'range':
#             if param['value_type'] == 'int':
#                 parameters.append(RangeParameter(name=param['name'], parameter_type=ParameterType.INT, lower=param['bounds'][0], upper=param['bounds'][1]))
#             else:
#                 parameters.append(RangeParameter(name=param['name'], parameter_type=ParameterType.FLOAT, lower=param['bounds'][0], upper=param['bounds'][1]))
#         elif param['type'] == 'fixed':
#             if param['value_type'] == 'int':
#                 parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.INT, value=param.value))
#             elif param['value_type'] == 'str':
#                 parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.STRING, value=param.value))
#             elif param['value_type'] == 'bool':
#                 parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.BOOL, value=param.value))
#             else:
#                 parameters.append(FixedParameter(name=param.name, parameter_type=ParameterType.FLOAT, value=param.value))
#         elif param['type'] == 'choice':
#             parameters.append(ChoiceParameter(name=param.name, values=param.values))
#         else:
#             raise ValueError('The parameter type is not recognized')
#     return SearchSpace(parameters=parameters)


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

# class MockJobQueueClient:
#         """Dummy class to represent a job queue where the Ax `Scheduler` will
#         deploy trial evaluation runs during optimization.
#         """

#         jobs: Dict[str, MockJob] = {}

#         def __init__(self, agents =  None, pool = None, tmp_dir = None, parallel_agents = True):
#             self.agents = agents
#             self.pool = pool
#             self.tmp_dir = tmp_dir
#             self.parallel_agents = parallel_agents

#         def schedule_job_with_parameters(
#             self, parameters: Dict[str, Union[str, float, int, bool]]
#         ) -> int:
#             """Schedules an evaluation job with given parameters and returns job ID."""
#             # Code to actually schedule the job and produce an ID would go here;
#             job_id = str(uuid.uuid4())
#             mock = MockJob(job_id, parameters)
#             # add mock run to the queue q 
#             self.jobs[job_id] = MockJob(job_id, parameters)
#             self.pool.apply_async(self.jobs[job_id].run, args=(job_id, parameters, self.agents, self.tmp_dir, self.parallel_agents))

#             return job_id

#         def get_job_status(self, job_id: str) -> TrialStatus:
#             """ "Get status of the job by a given ID. For simplicity of the example,
#             return an Ax `TrialStatus`.
#             """
#             job = self.jobs[job_id]
#             # check if job_id.json exists in the tmp directory
#             if os.path.exists(os.path.join(self.tmp_dir,str(job_id)+'.json')):
#                 #load the results
#                 with open(os.path.join(self.tmp_dir,str(job_id)+'.json'), 'r') as fp:
#                     res_dic = json.load(fp)

#                 # check is nan in res_dic
#                 for key in res_dic.keys():
#                     if np.isnan(res_dic[key]):
#                         return TrialStatus.FAILED
                    
#                 return TrialStatus.COMPLETED
#             else:
#                 return TrialStatus.RUNNING

#         def get_outcome_value_for_completed_job(self, job_id: int) -> Dict[str, float]:
#             """Get evaluation results for a given completed job."""
#             job = self.jobs[job_id]
#             # In a real external system, this would retrieve real relevant outcomes and
#             # not a synthetic function value.
#             # check if job_id.json exists in the tmp directory
#             if os.path.exists(os.path.join(self.tmp_dir,str(job_id)+'.json')):
#                 #load the results
#                 with open(os.path.join(self.tmp_dir,str(job_id)+'.json'), 'r') as fp:
#                     res_dic = json.load(fp)
#                 # delete file
#                 # os.remove(os.path.join(self.tmp_dir,str(job_id)+'.json'))
#                 # print('WE ARE DELETING THE FILE')
#                 return res_dic
#             else:
#                 raise ValueError('The job is not completed yet')



# def get_mock_job_queue_client(MOCK_JOB_QUEUE_CLIENT) -> MockJobQueueClient:
#         """Obtain the singleton job queue instance."""
#         return MOCK_JOB_QUEUE_CLIENT


# class MockJobRunner(Runner):  # Deploys trials to external system.

#     def __init__(self, agents = None, pool = None, tmp_dir = None, parallel_agents = True):
#         self.agents = agents
#         self.pool = pool
#         self.tmp_dir = tmp_dir
#         self.parallel_agents = parallel_agents
#         self.MOCK_JOB_QUEUE_CLIENT = MockJobQueueClient(agents = self.agents, pool = self.pool, tmp_dir = self.tmp_dir, parallel_agents = self.parallel_agents)

#     def _get_mock_job_queue_client(self) -> MockJobQueueClient:
#         """Obtain the singleton job queue instance."""
#         return self.MOCK_JOB_QUEUE_CLIENT
    
#     def run(self, trial: BaseTrial) -> Dict[str, Any]:
#         """Deploys a trial based on custom runner subclass implementation.

#         Args:
#             trial: The trial to deploy.

#         Returns:
#             Dict of run metadata from the deployment process.
#         """
#         if not isinstance(trial, Trial) and not isinstance(trial, BatchTrial):
#             raise ValueError("This runner only handles `Trial`.")

#         mock_job_queue = self._get_mock_job_queue_client()

#         run_metadata = []
#         if isinstance(trial, BatchTrial):
#             for arm in trial.arms:
#                 job_id = mock_job_queue.schedule_job_with_parameters(
#                     parameters=arm.parameters
#                 )
#                 # This run metadata will be attached to trial as `trial.run_metadata`
#                 # by the base `Scheduler`.
#                 arm.run_metadata = {"job_id": job_id}
#         else:
#             job_id = mock_job_queue.schedule_job_with_parameters(
#                 parameters=trial.arm.parameters
#             )

#         # This run metadata will be attached to trial as `trial.run_metadata`
#         # by the base `Scheduler`.
#         return {"job_id": job_id}

#     def poll_trial_status(
#         self, trials: Iterable[BaseTrial]
#     ) -> Dict[TrialStatus, Set[int]]:
#         """Checks the status of any non-terminal trials and returns their
#         indices as a mapping from TrialStatus to a list of indices. Required
#         for runners used with Ax ``Scheduler``.

#         NOTE: Does not need to handle waiting between polling calls while trials
#         are running; this function should just perform a single poll.

#         Args:
#             trials: Trials to poll.

#         Returns:
#             A dictionary mapping TrialStatus to a list of trial indices that have
#             the respective status at the time of the polling. This does not need to
#             include trials that at the time of polling already have a terminal
#             (ABANDONED, FAILED, COMPLETED) status (but it may).
#         """
#         status_dict = defaultdict(set)
#         for trial in trials:
#             mock_job_queue = self._get_mock_job_queue_client()
#             status = mock_job_queue.get_job_status(
#                 job_id=trial.run_metadata.get("job_id")
#             )
#             status_dict[status].add(trial.index)

#         return status_dict
    
# class BraninForMockJobMetric(Metric):  # Pulls data for trial from external system.
#     def __init__(self, name = None, agents = None, pool = None, tmp_dir = None, parallel_agents = True, **kwargs):
#         self.agents = agents
#         self.pool = pool
#         self.tmp_dir = tmp_dir
#         self.parallel_agents = parallel_agents
#         self.MOCK_JOB_QUEUE_CLIENT = MockJobQueueClient(agents = self.agents, pool = self.pool, tmp_dir = self.tmp_dir, parallel_agents = self.parallel_agents)
#         super().__init__(name=name, **kwargs)

#     def _get_mock_job_queue_client(self) -> MockJobQueueClient:
#         """Obtain the singleton job queue instance."""
#         return self.MOCK_JOB_QUEUE_CLIENT

#     def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:
#         """Obtains data via fetching it from ` for a given trial."""
#         if not isinstance(trial, Trial) and not isinstance(trial, BatchTrial):
#             raise ValueError("This metric only handles `Trial`.")

#         try:
#             mock_job_queue = self._get_mock_job_queue_client()

#             # Here we leverage the "job_id" metadata created by `MockJobRunner.run`.
#             if isinstance(trial, BatchTrial):
#                 lst_df_dict = []
#                 for arm in trial.arms:
#                     job_id = arm.run_metadata.get("job_id")
#                     while not os.path.exists(os.path.join(self.tmp_dir,str(job_id)+'.json')):
#                         time.sleep(.1)

#                     # branin_data = mock_job_queue.get_outcome_value_for_completed_job(
#                     #     job_id=trial.run_metadata.get("job_id")
#                     # )
#                     # arm.run_metadata.get("job_id")
#                     branin_data = mock_job_queue.get_outcome_value_for_completed_job(
#                         job_id=arm.run_metadata.get("job_id")
#                     )

#                     name_ = list(branin_data.keys())[0]

#                     df_dict = {
#                         "trial_index": trial.index,
#                         "metric_name": self.name,
#                         "arm_name": arm.name,
#                         "mean": branin_data.get(self.name),
#                         # Can be set to 0.0 if function is known to be noiseless
#                         # or to an actual value when SEM is known. Setting SEM to
#                         # `None` results in Ax assuming unknown noise and inferring
#                         # noise level from data.
#                         "sem": None,
#                     }
#                     lst_df_dict.append(df_dict)
#                 return Ok(value=Data(df=pd.DataFrame.from_records(lst_df_dict)))
#             else:

#                 job_id = trial.run_metadata.get("job_id")
#                 while not os.path.exists(os.path.join(self.tmp_dir,str(job_id)+'.json')):
#                     time.sleep(.1)
#                 branin_data = mock_job_queue.get_outcome_value_for_completed_job(
#                         job_id=arm.run_metadata.get("job_id")
#                     )
#                 name_ = list(branin_data.keys())[0]
#                 df_dict = {
#                     "trial_index": trial.index,
#                     "metric_name": self.name,
#                     "arm_name": arm.name,
#                     "mean": branin_data.get(self.name),
#                     # Can be set to 0.0 if function is known to be noiseless
#                     # or to an actual value when SEM is known. Setting SEM to
#                     # `None` results in Ax assuming unknown noise and inferring
#                     # noise level from data.
#                     "sem": None,
#                 }
#                 return Ok(value=Data(df=pd.DataFrame.from_records([df_dict])))
#         except Exception as e:
#             return Err(
#                 MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
#             )

from optimpv.axBOtorch.axBOtorchOptimizer import axBOtorchOptimizer
from optimpv.axBOtorch.axUtils import *
from optimpv.axBOtorch.axSchedulerUtils import *
class axBOtorchOptimizer_runner(axBOtorchOptimizer):
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
     
    def _optimize(self):
        # from kwargs
        enforce_sequential_optimization = self.kwargs.get('enforce_sequential_optimization',False)
        global_max_parallelism = self.kwargs.get('global_max_parallelism',-1)
        verbose_logging = self.kwargs.get('verbose_logging',False)
        global_stopping_strategy = self.kwargs.get('global_stopping_strategy',None)
        outcome_constraints = self.kwargs.get('outcome_constraints',None)
        parameter_constraints = self.kwargs.get('parameter_constraints',None)
        parallel_agents = self.kwargs.get('parallel_agents',True)
        max_number_cores = self.kwargs.get('max_number_cores',-1)
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
            self.ax_client = AxClient(generation_strategy=gs, enforce_sequential_optimization=enforce_sequential_optimization)
        
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
        
        while n < total_trials:
            # check the current batch size
            curr_batch_size = self.batch_size[np.argmax(n_step_points>n)]

            # Create a new scheduler for each batch with the current batch size
            scheduler = Scheduler(
                experiment=self.ax_client.experiment,
                generation_strategy=self.ax_client.generation_strategy,
                options=SchedulerOptions(run_trials_in_batches=True,init_seconds_between_polls=0.1,trial_type=TrialType.BATCH_TRIAL,batch_size=curr_batch_size,logging_level=0),
            )

            n += curr_batch_size
            # i = 0
            if n > total_trials:
                curr_batch_size = curr_batch_size - (n-total_trials)
            
            scheduler.run_n_trials(max_trials=1)

            # i += 1

        q.close()
        q.join()
    


if __name__ == '__main__':
    pass