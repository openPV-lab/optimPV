"""axBOtorchOptimizer module. This module contains the axBOtorchOptimizer class. The class is used to run the bayesian optimization process using the Ax library."""
######### Package Imports #########################################################################

import os, re, torch, math, copy, logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from logging import Logger
from joblib import Parallel, delayed
from functools import partial

from torch.quasirandom import SobolEngine
from torch.multiprocessing import Pool, set_start_method

from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize, normalize
from botorch.models import SingleTaskGP

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from ax import *
from ax.api.client import Client
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.transition_criterion import MinTrials
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.adapter.factory import Generators
from ax.api.protocols.metric import IMetric
from ax.utils.common.logger import set_ax_logger_levels

from optimpv import *
from optimpv.optimizers.axBOtorch.axUtils import *
from optimpv.optimizers.axBOtorch.TuRBOGenerationNode import TuRBOGenerationNode
from optimpv.optimizers.axBOtorch.TuRBOGenerationNode import TuRBOGlobalStoppingStrategy
from optimpv.optimizers.axBOtorch.MorboGenerationNode import MorboGenerationNode
from optimpv.optimizers.axBOtorch.REITuRBOGenerationNode import REITuRBOGenerationNode
from optimpv.optimizers.axBOtorch.CasmopolitanGenerationNode import CasmopolitanGenerationNode
from optimpv.general.logger import get_logger, _round_floats_for_logging
from optimpv.general.BaseAgent import BaseAgent

# try: # needed for multiprocessing when using pytorch
set_start_method('spawn',force=True)
# except RuntimeError:
#     print("spawn method already set")
#     pass

# Logger setup
set_ax_logger_levels(logging.WARN)
logger: Logger = get_logger('axBOtorchOptimizer')
ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)

######### Optimizer Definition #######################################################################
class axBOtorchOptimizer(BaseAgent):
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
    existing_data : DataFrame, optional
        existing data to use for the optimization process, by default None
    suggest_only : bool, optional
        if True, the optimization process will only suggest new points without running the agents, by default False
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
    def __init__(self, params = None, agents = None, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,10], batch_size = [10,2], ax_client = None,  max_parallelism = -1,model_kwargs_list = None, model_gen_kwargs_list = None, existing_data = None, suggest_only = False, name = 'ax_opti', **kwargs):
               
        self.params = params
        if not isinstance(agents, list):
            agents = [agents]
        self.agents = agents
        for agent in self.agents: # make sure that the agents have the same params as the optimizer
            agent.params = self.params
        self.models = models
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.all_metrics = None
        self.all_metrics,self.all_minimize = self.create_metrics_list()
        self.all_tracking_metrics = None
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
        self.existing_data = existing_data
        self.suggest_only = suggest_only
        
        if len(n_batches) != len(models):
            raise ValueError('n_batches and models must have the same length')
        if type(batch_size) == int:
            self.batch_size = [batch_size]*len(models)
        if len(batch_size) != len(models):
            raise ValueError('batch_size and models must have the same length')
        
        self.torch_dtype = self.kwargs.get('torch_dtype',torch.float64)
        torch.set_default_dtype(self.torch_dtype)


    def create_generation_strategy(self,skip_center=False,skip_sobol=False):
        """ Create a generation strategy for the optimization process using the models and the number of batches and batch sizes. See ax documentation for more details: https://ax.dev/tutorials/generation_strategy.html

        Returns
        -------
        GenerationStrategy
            The generation strategy for the optimization process
        skip_center : bool, optional
            If True, the Center model will be skipped, by default False
        skip_sobol : bool, optional
            If True, the Sobol model will be skipped, by default False

        Raises
        ------
        ValueError
            If the model is not a string or a Models enum
        """        

        nodes_list = []
        # get all names
        names = []
        Gen_strat_name = ""
        generators = []
        for i, model in enumerate(self.models):
            if type(model) == str and model.lower() == 'center':
                node_name = 'Center'
                Gen_strat_name += 'Center+'
                generators.append(node_name)
                names.append(node_name)
                continue

            if type(model) == str and model.lower() == 'turbo':
                node_name = 'TuRBO'
                Gen_strat_name += 'TuRBO'
                generators.append(node_name)
                names.append(node_name)
                continue

            if type(model) == str and model.lower() == 'morbo':
                node_name = 'MORBO'
                Gen_strat_name += 'MORBO'
                generators.append(node_name)
                names.append(node_name)
                continue

            if type(model) == str and model.lower() in ('rei_turbo', 'reiturbo'):
                node_name = 'REITuRBO'
                Gen_strat_name += 'REITuRBO'
                generators.append(node_name)
                names.append(node_name)
                continue

            if type(model) == str and model.lower() == 'casmopolitan':
                node_name = 'Casmopolitan'
                Gen_strat_name += 'Casmopolitan'
                generators.append(node_name)
                names.append(node_name)
                continue
                
            if type(model) == str:
                node_name = model
                model = Generators[model]
                generators.append(model)

            elif isinstance(model, Generators):
                model = model
                node_name = model.__name__
                generators.append(model)
            else:
                raise ValueError('Model must be a string or a Models enum')
            Gen_strat_name += node_name + '+'
            names.append(node_name)

        # remove the last +
        Gen_strat_name = Gen_strat_name[:-1]
        # Create the generator spec
        for i, model in enumerate(generators):
            # Get the next node name
            if skip_center and names[i].lower() == 'center':
                continue
            if skip_sobol and names[i].lower() == 'sobol':
                continue

            if names[i].lower() == 'center':
                # Center node is a customized node that uses a simplified logic and has a
                # built-in transition criteria that transitions after generating once.
                nodes_list.append(CenterGenerationNode(next_node_name=names[i+1]))
                continue
            
            if names[i].lower() == 'turbo':
                # Turbo node is a customized node 
                objective = self.create_objectives()
                if "," in objective:  # Multiple objectives in string format
                    raise ValueError('TuRBOGenerationNode does not support multiple objectives')
                num_free_params = len([p for p in self.params if p.type != 'fixed'])
                minimize = objective.startswith('-')
                acq = self.model_kwargs_list[i].get('acq', 'ei')
                acqf_kwargs = self.model_kwargs_list[i].get('acqf_kwargs', {})
                if acqf_kwargs is None or acqf_kwargs == {}:
                    if acq == 'ei':
                        acqf_kwargs = {"raw_samples": 512, "num_restarts": 10}
                    elif acq == 'ts':
                        acqf_kwargs = {"n_candidates": min(10000, max(2000, 200 * num_free_params))}
                tkwargs = {}
                if self.model_kwargs_list[i].get("torch_device") is not None:
                    tkwargs["device"] = self.model_kwargs_list[i].get("torch_device")
                else:
                    tkwargs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.model_kwargs_list[i].get("torch_dtype") is not None:
                    tkwargs["dtype"] = self.model_kwargs_list[i].get("torch_dtype")
                nodes_list.append(TuRBOGenerationNode(name=names[i], 
                                                      model_options=self.model_kwargs_list[i],
                                                      batch_size=self.batch_size[i],
                                                      acqf=acq,
                                                      acqf_kwargs=acqf_kwargs,
                                                      **tkwargs,
                                                      maximize=not minimize,
                                                      ))

                continue

            if names[i].lower() == 'morbo':
                objective = self.create_objectives()
                if "," not in objective:  # Multiple objectives in string format
                    raise ValueError('MorboGenerationNode only supports multiple objectives')
               
                tkwargs = {}
                if self.model_kwargs_list[i].get("torch_device") is not None:
                    tkwargs["device"] = self.model_kwargs_list[i].get("torch_device")
                else:
                    tkwargs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.model_kwargs_list[i].get("torch_dtype") is not None:
                    tkwargs["dtype"] = self.model_kwargs_list[i].get("torch_dtype")
                nodes_list.append(MorboGenerationNode(name=names[i], 
                                                      model_options=self.model_kwargs_list[i],
                                                      batch_size=self.batch_size[i],
                                                      **tkwargs,
                                                      ))

                continue
            
            if names[i].lower() == 'reiturbo':
                objective = self.create_objectives()
                if "," in objective:
                    raise ValueError('REITuRBOGenerationNode does not support multiple objectives')

                minimize = objective.startswith('-')
                acq = self.model_kwargs_list[i].get('acq', 'ts')
                
                region_init_points = self.model_kwargs_list[i].get("region_init_points")
                if region_init_points % self.batch_size[i] != 0:
                    raise ValueError("region_init_points must be divisible by batch_size. "
                                     "Otherwise, the initial seeding and restart logic of the TRs will not work properly.")
                
                tkwargs = {}
                if self.model_kwargs_list[i].get("torch_device") is not None:
                    tkwargs["device"] = self.model_kwargs_list[i].get("torch_device")
                else:
                    tkwargs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.model_kwargs_list[i].get("torch_dtype") is not None:
                    tkwargs["dtype"] = self.model_kwargs_list[i].get("torch_dtype")

                # REI-TuRBO uses REI for region seeding and TS/EI for local trust-region steps.
                nodes_list.append(
                    REITuRBOGenerationNode(
                        name=names[i],
                        model_options=self.model_kwargs_list[i],
                        batch_size=self.batch_size[i],
                        acqf=acq,
                        **tkwargs,
                        maximize=not minimize,
                    )
                )
                
                continue

            if names[i].lower() == 'casmopolitan':
                objective = self.create_objectives()
                if "," in objective:
                    raise ValueError('CasmopolitanGenerationNode does not support multiple objectives')
                minimize = objective.startswith('-')

                nodes_list.append(
                    CasmopolitanGenerationNode(
                        name=names[i],
                        model_options=self.model_kwargs_list[i],
                        batch_size=self.batch_size[i],
                        maximize=not minimize,
                    )
                )

                continue

            # Create the generator spec
            generator_spec = GeneratorSpec(
                                            generator_enum=model,
                                            model_kwargs=self.model_kwargs_list[i],
                                            # We can specify various options for the optimizer here.
                                            model_gen_kwargs = self.model_gen_kwargs_list[i], 
            )

            # Create the generation node
            if i < len(self.models)-1:
                node = GenerationNode(
                    name=names[i],
                    generator_specs=[generator_spec],
                    transition_criteria=[
                        MinTrials(threshold=self.n_batches[i]*self.batch_size[i],
                                  transition_to=names[i+1],
                        )
                    ],
                )
            else:
                node = GenerationNode(
                    name=names[i],
                    generator_specs=[generator_spec],
                )

            nodes_list.append(node)
        if len(nodes_list) == 0:
            raise ValueError('No generation nodes were created. Please check the models and skip_center and skip_sobol parameters.')
        # Create the generation strategy
        return GenerationStrategy(
            name=Gen_strat_name,
            nodes=nodes_list
        )

    def get_tracking_metrics(self, agents):
        """ Extract tracking metrics from agents
        
        Parameters
        ----------
        agents : list
            List of Agent objects
            
        Returns
        -------
        list
            List of tracking metric names formatted with agent name prefix
        """
        tracking_metrics = []
        for agent in agents:
            if hasattr(agent, 'all_agent_tracking_metrics') :
                if agent.all_agent_tracking_metrics is not None:
                    for metric in agent.all_agent_tracking_metrics:
                        tracking_metrics.append(metric)

        return tracking_metrics

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
        
        objectives = ""

        # objectives = {}
        for agent in self.agents:
            for i in range(len(agent.all_agent_metrics)):
                if agent.minimize[i]:
                    objectives += "-"+agent.all_agent_metrics[i] + ","
                    if append_metrics:
                        self.all_metrics.append(agent.all_agent_metrics[i])
                else:
                    objectives += agent.all_agent_metrics[i] + ","
                    if append_metrics:
                        self.all_metrics.append(agent.all_agent_metrics[i])

        # remove the last comma
        objectives = objectives[:-1]

        return objectives
    
    def attach_existing_data(self):
        """ Attach existing data to the Ax client
        """

        if self.existing_data is not None:
            # Convert the existing data to a DataFrame if it is not already
            if not isinstance(self.existing_data, pd.DataFrame):
                raise ValueError("existing_data must be a pandas DataFrame")

        # rescale the existing data to match what is expected by the Ax client
        copy_data = self.descale_dataframe(copy.deepcopy(self.existing_data), self.params)
        
        for index, row in copy_data.iterrows():
            # Create a parameterization dictionary
            # parameterization = {p.name: copy_data[p.name].iloc[index] for p in self.params if p.type != 'fixed'}
            parameterization = {}
            for p in self.params:
                if p.type != 'fixed':
                    if p.value_type == 'float':
                        parameterization[p.name] = copy_data[p.name].iloc[index]
                    elif p.value_type == 'int':
                        parameterization[p.name] = int(copy_data[p.name].iloc[index])
                    elif p.value_type == 'str':
                        parameterization[p.name] = str(copy_data[p.name].iloc[index])
                    elif p.value_type == 'cat' or p.value_type == 'sub':
                        parameterization[p.name] = str(copy_data[p.name].iloc[index])

            # If the parameterization is empty, skip this row
            if not parameterization:
                continue

            # Create a raw data dictionary and keep the correct types
            raw_data = {metric: copy_data[metric].iloc[index] for metric in self.all_metrics + self.all_tracking_metrics if metric in row}

            # If the raw data is empty, skip this row
            if not raw_data:
                continue

            trial_index = self.ax_client.attach_trial(
                parameters=parameterization,
            )
            # Then complete the trial with the existing data
            self.ax_client.complete_trial(
                trial_index=trial_index, raw_data=raw_data,
            )
            if self.kwargs.get('verbose_logging', False):
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info(f"Attached trial {trial_index} with parameters {parameterization} and raw data {raw_data}")

    def get_initial_data_from_existing_data_turbo(self):

        if self.existing_data is not None:
            if not isinstance(self.existing_data, pd.DataFrame):
                raise ValueError("existing_data must be a pandas DataFrame")

        # rescale the existing data to match what is expected
        copy_data = self.descale_dataframe(copy.deepcopy(self.existing_data), self.params)
        X_turbo, Y_turbo, Y_tracking = [], [], []
        X_turbo = copy_data[[p.name for p in self.params if p.type != 'fixed']].values
        Y_turbo = copy_data[self.all_metrics].values
        Y_tracking = copy_data[self.all_tracking_metrics].values if len(self.all_tracking_metrics) > 0 else None

        return X_turbo, Y_turbo, Y_tracking

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
    
    def optimize(self):
        """ Run the optimization process using the agents and the parameters. The optimization process uses the Ax library. The optimization process runs the agents in parallel if the parallel attribute is True. The optimization process runs using the parameters, agents, models, n_batches, batch_size, max_parallelism, model_kwargs_list, model_gen_kwargs_list, name and kwargs attributes of the class.

        Raises
        ------
        ValueError
            If the number of batches and the number of models are not the same

        """  

        self.optimize_sequential()
        #Note: I might reimplement the runner option in a future version, but for now it is not used.

    def optimize_sequential(self):
        """ Run the optimization process using the agents and the parameters. The optimization process uses the Ax library. The optimization process runs the agents in parallel if the parallel attribute is True. The optimization process runs using the parameters, agents, models, n_batches, batch_size, max_parallelism, model_kwargs_list, model_gen_kwargs_list, name and kwargs attributes of the class.

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
        parallel = self.kwargs.get('parallel',True)
        

        # if len(self.agents) == 1: # If there is only one agent, disable parallelism
        #     parallel_agents = False

        # create parameters space from params
        parameters_space, fixed_parameters = ConvertParamsAx(self.params)

        # Get tracking metrics directly
        self.all_tracking_metrics = self.get_tracking_metrics(self.agents)

        # # create generation strategy
        gs = self.create_generation_strategy()

        # create ax client
        if self.ax_client is None:
            self.ax_client = Client()

        # Configure the experiment
        self.ax_client.configure_experiment(
            name=self.name,
            parameters=parameters_space,
            parameter_constraints=parameter_constraints,
            )
        
        objective = self.create_objectives()
        
        self.ax_client.configure_optimization(objective=objective)

        if len(self.all_tracking_metrics) != 0:
            self.ax_client.configure_metrics([IMetric(name=m) for m in self.all_tracking_metrics])

        if self.existing_data is not None:
            self.attach_existing_data()
        
        self.ax_client.set_generation_strategy(generation_strategy=gs)
        
        # run optimization
        num = 0
        total_trials = sum(np.asarray(self.n_batches)*np.asarray(self.batch_size))
        n_step_points = np.cumsum(np.asarray(self.n_batches)*np.asarray(self.batch_size))
        size_pool = None
        if self.suggest_only and not(self.models[0] != 'SOBOL' and self.existing_data is None):
            # If suggest_only is True, we only suggest the trials without running the agents
            trials = self.ax_client.get_next_trials(total_trials)
            
            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info(f"Suggesting {total_trials} trials without running the agents.")

            return

        while num < total_trials:
            # check the current batch size
            curr_batch_size = self.batch_size[np.argmax(n_step_points>num)]
            num += curr_batch_size
            if num > total_trials:
                curr_batch_size = curr_batch_size - (num-total_trials)

            # parameters, trial_index = self.ax_client.get_next_trials(curr_batch_size)
            trials = self.ax_client.get_next_trials(curr_batch_size)
            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                for i in trials.keys():
                    logger.info(f"Trial {i} with parameters: {trials[i]}")

            
            trials_index = list(trials.keys())
            parameters = [] * len(trials)
            for i in trials_index:
                parameters.append(trials[i])

            if parallel:
                all_results = Parallel(
                    n_jobs=min(len(parameters) * len(self.agents), self.max_parallelism)
                )(
                    delayed(lambda ag, p, pi: (pi, ag.run_Ax(p)))(
                        agent, p, pi
                    )
                    for agent in self.agents
                    for pi, p in enumerate(parameters)
                )

                # merge results
                main_results = [{} for _ in parameters]
                for param_idx, res in all_results:
                    main_results[param_idx].update(res)
            else:
                main_results = [{} for _ in parameters]
                for agent in self.agents:
                    for pi, p in enumerate(parameters):
                        res = agent.run_Ax(p)
                        main_results[pi].update(res)

            idx = 0
            for trial_index_, raw_data in zip(trials_index, main_results):
                got_nan = False
                for key in raw_data.keys():
                    if np.isnan(raw_data[key]):
                        got_nan = True
                        break
                if not got_nan:
                    if verbose_logging:
                        logging_level = 20
                        logger.setLevel(logging_level)
                        logger.info(f"Trial {trial_index_} completed with results: {raw_data}")# and parameters: {parameters[idx]}")
                    self.ax_client.complete_trial(trial_index_, raw_data=raw_data)
                else:
                    if verbose_logging:
                        logging_level = 20
                        logger.setLevel(logging_level)
                        logger.info(f"Trial {trial_index_} failed with results: {raw_data} and parameters: {parameters[idx]}")
                    self.ax_client.mark_trial_failed(trial_index_)
                idx += 1

            # check global stopping strategy
            if global_stopping_strategy is not None:
                if isinstance(global_stopping_strategy, TuRBOGlobalStoppingStrategy):
                    stop, message = global_stopping_strategy.should_stop_optimization(self.ax_client._experiment,client=self.ax_client)
                else:
                    stop, message = global_stopping_strategy.should_stop_optimization(self.ax_client._experiment)
                if stop:
                    if verbose_logging:
                        logging_level = 20
                        logger.setLevel(logging_level)
                        logger.info(f"Global stopping strategy triggered: {message}")
                    break


    def update_params_with_best_balance(self,return_best_balance=False):
        """ Update the parameters with the best balance of all metrics. 
        The best balance is defined by ranking the results for each metric and taking the parameters that has the lowest sum of ranks.
        
        Raises
        ------
        ValueError
            We need at least one metric to update the parameters
        """        

        # if we have one objective
        if len(self.all_metrics) == 1:
            scaled_best_parameters = self.ax_client.get_best_parameterization(use_model_predictions=False)[0]
            self.params_w(scaled_best_parameters,self.params)
        # if we have multiple objectives
        elif len(self.all_metrics) > 1:
            # We do this because the ax_client.get_pareto_optimal_parameters does not necessarily return the best parameters for a balanced results on all objectives
            df = get_df_ax_client_metrics(self.params, self.ax_client, self.all_metrics)
            metrics = self.all_metrics
            minimizes_ = []

            for agent in self.agents:
                for i in range(len(agent.minimize)):
                    minimizes_.append(agent.minimize[i])

            # Filter out rows with NaN values in any metric
            df_filtered = df.dropna(subset=metrics)
            
            if len(df_filtered) == 0:
                raise ValueError('All rows contain NaN values in at least one metric')

            ranked_df = copy.deepcopy(df_filtered)
            ranks = []
            for i in range(len(metrics)):
                ranked_df[metrics[i]+'_rank'] = ranked_df[metrics[i]].rank(ascending=minimizes_[i])
                ranks.append(ranked_df[metrics[i]+'_rank'])
            # get the index of the best balance
            best_balance_index = np.argmin(np.sum(np.array(ranks), axis=0))

            # get the best parameters
            scaled_best_parameters = ranked_df.iloc[best_balance_index].to_dict()
            
            dum_dic = {}
            for p in self.params:
                if p.type != 'fixed':
                    dum_dic[p.name] = scaled_best_parameters[p.name]
                else:
                    dum_dic[p.name] = p.value
            scaled_best_parameters = dum_dic

            for p in self.params:
                if p.name in scaled_best_parameters.keys():
                    p.value = scaled_best_parameters[p.name]
            if return_best_balance:
                return best_balance_index, scaled_best_parameters
        else:
            raise ValueError('We need at least one metric to update the parameters')

    def optimize_turbo(self,acq_turbo='ts',force_continue = False, kwargs_turbo_state={},kwargs_turbo={}):
        """Run the optimzation using Turbo. This is based on the Botorch implementation of Turbo. See https://botorch.org/docs/tutorials/turbo_1/ for more details.

        Parameters
        ----------
        acq_turbo : str, optional
            The acquisition function to use can be 'ts' or 'ei', by default 'ts'
        force_continue : bool, optional
            If True, the optimization will continue even if a restart is triggered, by default False
        kwargs_turbo_state : dict, optional
            The kwargs to use for the TurboState, by default {}
            can be: 
            - length: float, by default 0.8
            - length_min: float, by default 0.5**7
            - length_max: float, by default 1.6
            - success_tolerance: int, by default 10
        kwargs_turbo : dict, optional
            The kwargs to use for the Turbo, by default {}

        Raises
        ------
        ValueError
            Turbo does not support outcome constraints
        ValueError
            Turbo only supports single objective optimization
        ValueError
            Turbo only supports 2 models
        ValueError
            Turbo only supports Sobol as the first model
        ValueError
            Turbo only supports BoTorch as the second model
        """            
        device = self.kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = self.kwargs.get('dtype', torch.float64)
        
        parameters_space, fixed_parameters = ConvertParamsAx(self.params)
        objective = self.create_objectives()

        # Get tracking metrics directly
        self.all_tracking_metrics = self.get_tracking_metrics(self.agents)

        # make sure that we do not take fixed params into account
        free_pnames = [p.name for p in parameters_space]
        
        dim = len(free_pnames)

        parallel = self.kwargs.get('parallel',True)
        verbose_logging = self.kwargs.get('verbose_logging',True)
        enforce_sequential_optimization = self.kwargs.get('enforce_sequential_optimization',False)
        global_stopping_strategy = self.kwargs.get('global_stopping_strategy',None)
        outcome_constraints = self.kwargs.get('outcome_constraints',None)
        parameter_constraints = self.kwargs.get('parameter_constraints',None)
        NUM_RESTARTS = kwargs_turbo.get('NUM_RESTARTS', 10)
        RAW_SAMPLES = kwargs_turbo.get('RAW_SAMPLES', 512)
        N_CANDIDATES = kwargs_turbo.get('N_CANDIDATES', min(5000, max(2000, 200 * dim)))

        if parameter_constraints is not None:
            inequality_constraints = _parse_inequality_constraints(
                parameter_constraints, free_pnames, device=device, dtype=dtype
            )
        else:
            inequality_constraints = None

        if outcome_constraints is not None:
            raise ValueError('Turbo does not support outcome constraints')  
        
        for par in self.params:
            if par.stepsize is not None:
                raise ValueError('Turbo does not support parameters with stepsize')
        
        # check if we have a single objective
        got_MOO = False
        if "," in objective:  # Multiple objectives in string format
            got_MOO = True
            logger.warning(r'//!\\ Turbo normally only supports single objective optimization, but multiple objectives were provided. The mean of the objectives will be used.')
            # check that all objectives are to be minimized or maximized
            obj_list = objective.split(',')
            first_minimize = obj_list[0].startswith('-')
            for obj in obj_list[1:]:
                if obj.startswith('-') != first_minimize:
                    raise ValueError('Turbo only supports single objective optimization with all objectives to be minimized or maximized')
            # raise ValueError('Turbo only supports single objective optimization')
        
        # check if we minimize (objective string starts with -)
        minimize = objective.startswith('-')
        if minimize:
            fac = -1
        else:
            fac = 1
        
        if self.suggest_only:
            if len(self.models)>1:
                raise ValueError('Turbo only supports 1 model in suggest_only mode')
            if self.models[0] != 'BOTORCH_MODULAR':
                raise ValueError('Turbo only supports BoTorch as the model in suggest_only mode')
        else:
            if len(self.models)>2:
                raise ValueError('Turbo only supports 2 models')
            if self.models[0] != 'SOBOL':
                raise ValueError('Turbo only supports Sobol as the first model')
            if self.models[1] != 'BOTORCH_MODULAR':
                raise ValueError('Turbo only supports BoTorch as the second model')
        
        # Set the device and dtype
        max_cholesky_size = float("inf")  # Always use Cholesky
        
        total_trials = sum(np.asarray(self.n_batches)*np.asarray(self.batch_size))
        if verbose_logging:
            logger.info('Starting optimization with %d batches and a total of %d trials',sum(np.asarray(self.n_batches)),total_trials)

        # Create the bounds for the parameters
        bounds = torch.tensor([p.bounds for p in parameters_space], device=device, dtype=dtype)
        bounds = bounds.transpose(0,1) # transpose bounds
        count_failure = 0
        if self.existing_data is not None:
            # If we have existing data, we need to use it to initialize the optimization
            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info('Using existing data for initialization')
            X_turbo, Y_turbo, Y_tracking = self.get_initial_data_from_existing_data_turbo()
            # convert to torch tensors
            X_turbo_un = torch.tensor(X_turbo, device=device, dtype=dtype)
            X_turbo = torch.tensor(X_turbo, device=device, dtype=dtype)
            #normalize the X_turbo
            X_turbo = normalize(X_turbo, bounds=bounds)
            Y_turbo = torch.tensor(Y_turbo, device=device, dtype=dtype)
            Y_tracking = torch.tensor(Y_tracking, device=device, dtype=dtype)
            
        else:
            # Start with a Sobol sequence
            n_total_sobol = self.n_batches[0]*self.batch_size[0]
            num_sobol = 0
            
            # Create and run initial points per batch
            count_batch = 1
            while num_sobol < n_total_sobol:
                if verbose_logging:
                    logging_level = 20
                    logger.setLevel(logging_level)
                    logger.info('Starting Sobol batch %d with %d trials', count_batch, self.batch_size[0])
                
                # Get initial points
                X_turbo = get_initial_points(
                    dim=dim,
                    n_pts=self.batch_size[0],
                    device=device,
                    dtype=dtype,
                    inequality_constraints=inequality_constraints,
                    bounds=bounds,
                )
                
                # unnormalize
                X_turbo_un = unnormalize(X_turbo, bounds=bounds)
                # build list of dicts with noam
                dics = []
                for p in X_turbo_un:
                    p = p.cpu().numpy()
                    # write p into a dict with the param names as keys if the param is not fixed
                    idx = 0
                    dum_dict = {}
                    for i in range(len(self.params)):
                        if self.params[i].type != 'fixed':
                            dum_dict[self.params[i].name] = p[idx]
                            idx += 1
                    dics.append(dum_dict)


                if parallel:
                    all_results = Parallel(
                        n_jobs=min(len(dics) * len(self.agents), self.max_parallelism)
                    )(
                        delayed(lambda ag, p, pi: (pi, ag.run_Ax(p)))(
                            agent, p, pi
                        )
                        for agent in self.agents
                        for pi, p in enumerate(dics)
                    )

                    # merge results
                    main_results = [{} for _ in dics]
                    for param_idx, res in all_results:
                        main_results[param_idx].update(res)
                else:
                    main_results = [{} for _ in dics]
                    for agent in self.agents:
                        for pi, p in enumerate(dics):
                            res = agent.run_Ax(p)
                            main_results[pi].update(res)

                # Only keep values from result dictionary that are in all_metrics
                if not got_MOO:
                    Y_turbo = torch.tensor([[res[self.all_metrics[0]]] for res in main_results], device=device, dtype=dtype)
                    Y_turbo_all = copy.deepcopy(Y_turbo)
                else:# take mean metric value
                    Y_turbo = torch.tensor([[np.mean([res[metric] for metric in self.all_metrics])] for res in main_results], device=device, dtype=dtype)
                    # do the same than tracking
                    tracking_data = [
                                [res.get(metric, float('nan')) for metric in self.all_metrics]
                                for res in main_results
                            ]
                    if tracking_data:
                        Y_turbo_all = torch.tensor(tracking_data, device=device, dtype=dtype)
                # multiplication factor
                Y_turbo = fac*Y_turbo

                # find idx where we have nan in Y_turbo
                nan_idx = torch.isnan(Y_turbo).any(dim=1)
                count_failure += nan_idx.sum().item()
                # remove nan from Y_turbo and X_turbo
                Y_turbo = Y_turbo[~nan_idx]
                Y_turbo_all = Y_turbo_all[~nan_idx]
                X_turbo = X_turbo[~nan_idx]

                # Also collect tracking metrics if they exist
                Y_tracking = None
                if self.all_tracking_metrics and len(self.all_tracking_metrics) > 0:
                    tracking_data = []
                    for res in main_results:
                        metrics_vals = []
                        for metric in self.all_tracking_metrics:
                            if metric in res:
                                metrics_vals.append(res[metric])
                            else:
                                metrics_vals.append(float('nan'))
                        tracking_data.append(metrics_vals)
                    if tracking_data:
                        Y_tracking = torch.tensor(tracking_data, device=device, dtype=dtype)
                    Y_tracking = Y_tracking[~nan_idx]
                num_sobol += self.batch_size[0]
                count_batch += 1
                
            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info('Finished Sobol with best value of %f', fac*max(Y_turbo).item())
        
        if not self.suggest_only:
            # Create a new state for each batch
            best_value = max(Y_turbo).item()
            state = TurboState(dim=dim, batch_size=self.batch_size[1], best_value=best_value,**kwargs_turbo_state)
            max_num_trials = self.n_batches[1]*self.batch_size[1]
            num_turbo = 0
            state = update_state(state=state, Y_next=Y_turbo)
            while (not num_turbo > max_num_trials) and not (state.restart_triggered and not force_continue):
                if verbose_logging:
                    logging_level = 20
                    logger.setLevel(logging_level)
                    if state.restart_triggered and force_continue:
                        logger.setLevel(logging_level)
                        logger.info('Restart triggered, but we force the optimization to continue.')
                try:
                    # Fit a GP model
                    train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()

                    try:
                        
                        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                            MaternKernel(
                                nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
                            )
                        )
                        model = SingleTaskGP(
                            X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
                        )
                        mll = ExactMarginalLogLikelihood(model.likelihood, model)
                        
                        # Do the fitting and acquisition function optimization inside the Cholesky context
                        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                            # Fit the model
                            fit_gpytorch_mll(mll)

                            # Create a batch
                            X_next = generate_batch(
                                state=state,
                                model=model,
                                X=X_turbo,
                                Y=train_Y,
                                batch_size=state.batch_size,
                                n_candidates=N_CANDIDATES,
                                num_restarts=NUM_RESTARTS,
                                raw_samples=RAW_SAMPLES,
                                acqf=acq_turbo,
                                device=device,
                                dtype=dtype,
                                minimize=minimize,
                                inequality_constraints=inequality_constraints,
                                bounds=bounds
                            )
                    except Exception as e:

                        logging_level = 20
                        logger.setLevel(logging_level)
                        logger.error(f"Error in Turbo batch {count_batch}: {e}")
                        logger.error(f"We are stopping the optimization process")
                        break

                    # Evaluate the batch
                    X_next_un = unnormalize(X_next, bounds=bounds)
                    # build list of dicts with noam
                    dics = []
                    for p in X_next_un:
                        p = p.cpu().numpy()
                        idx = 0
                        dum_dict = {}
                        for i in range(len(self.params)):
                            if self.params[i].type != 'fixed':
                                dum_dict[self.params[i].name] = p[idx]
                                idx += 1
                        dics.append(dum_dict)

                    # run agents
                    if parallel:
                        all_results = Parallel(
                            n_jobs=min(len(dics) * len(self.agents), self.max_parallelism)
                        )(
                            delayed(lambda ag, p, pi: (pi, ag.run_Ax(p)))(
                                agent, p, pi
                            )
                            for agent in self.agents
                            for pi, p in enumerate(dics)
                        )

                        # merge results
                        main_results = [{} for _ in dics]
                        for param_idx, res in all_results:
                            main_results[param_idx].update(res)
                    else:
                        main_results = [{} for _ in dics]
                        for agent in self.agents:
                            for pi, p in enumerate(dics):
                                res = agent.run_Ax(p)
                                main_results[pi].update(res)

                    # Only keep values from result dictionary that are in all_metrics
                    if not got_MOO:
                        Y_next = torch.tensor([[res[self.all_metrics[0]]] for res in main_results], device=device, dtype=dtype)
                        Y_next_all = copy.deepcopy(Y_next)
                    else: # take mean metric value
                        Y_next = torch.tensor([[np.mean([res[metric] for metric in self.all_metrics])] for res in main_results], device=device, dtype=dtype)
                        tracking_data = [
                                    [res.get(metric, float('nan')) for metric in self.all_metrics]
                                    for res in main_results
                                ]
                        if tracking_data:
                            Y_next_all = torch.tensor(tracking_data, device=device, dtype=dtype)
                            # Y_next_all = torch.tensor([[[res[metric] for metric in self.all_metrics]] for res in main_results], device=device, dtype=dtype)
                    # multiplication factor
                    Y_next = fac*Y_next

                    # find idx where we have nan in Y_turbo
                    nan_idx = torch.isnan(Y_next).any(dim=1)
                    count_failure += nan_idx.sum().item()
                    # remove nan from Y_next and X_next
                    Y_next = Y_next[~nan_idx]
                    Y_next_all = Y_next_all[~nan_idx]
                    X_next = X_next[~nan_idx]
                    
                    if Y_next.shape[0] != 0: # if we have at least one non-nan value
                        if nan_idx.sum() > state.batch_size:
                            raise ValueError("Too many NaN values in Y_next")
                        
                        # Also collect tracking metrics if they exist
                        Y_next_tracking = None
                        if self.all_tracking_metrics and len(self.all_tracking_metrics) > 0:
                            tracking_data = []
                            for res in main_results:
                                metrics_vals = []
                                for metric in self.all_tracking_metrics:
                                    if metric in res:
                                        metrics_vals.append(res[metric])
                                    else:
                                        metrics_vals.append(float('nan'))
                                tracking_data.append(metrics_vals)
                            if tracking_data:
                                Y_next_tracking = torch.tensor(tracking_data, device=device, dtype=dtype)
                            Y_next_tracking = Y_next_tracking[~nan_idx]
                        # Update state
                        state = update_state(state=state, Y_next=Y_next)

                        # Append data
                        X_turbo = torch.cat((X_turbo, X_next), dim=0)
                        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)
                        Y_turbo_all = torch.cat((Y_turbo_all, Y_next_all), dim=0)
                        if Y_tracking is not None and Y_next_tracking is not None:
                            Y_tracking = torch.cat((Y_tracking, Y_next_tracking), dim=0)
                        elif Y_next_tracking is not None:
                            Y_tracking = Y_next_tracking
                        
                    num_turbo += state.batch_size
                    
                    # Print current status
                    if verbose_logging:
                        logging_level = 20
                        logger.setLevel(logging_level)
                        logger.info(f"Finished Turbo batch {count_batch} with {state.batch_size} trials with current best value: {fac*state.best_value:.4e}, TR length: {state.length:.2e}")
                    
                    count_batch += 1
                except Exception as e:
                    logging_level = 20
                    logger.setLevel(logging_level)
                    logger.error(f"Error in Turbo batch {count_batch}: {e}")
                    logger.error(f"We are stopping the optimization process")
                    break
        else:
            # If suggest_only is True, we only suggest the trials without running the agents
            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info(f"Suggesting {total_trials} trials without running the agents.")

             # Create a new state for each batch
            # get previous best value from kwargs_turbo
            if 'best_value' in kwargs_turbo:
                best_value = kwargs_turbo['best_value']
            else:
                raise ValueError('best_value from the previous state must be provided in kwargs_turbo for suggest_only mode')
            
            # best_value = max(Y_turbo).item()
            state = TurboState(dim=dim, batch_size=self.batch_size[0], best_value=best_value,**kwargs_turbo_state)
            state = update_state(state=state, Y_next=Y_turbo)
            max_num_trials = self.n_batches[0]*self.batch_size[0] # when suggest_only is True, we only use the first batch size
            num_turbo = 0
             # Fit a GP model
            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
            Y_turbo_all = copy.deepcopy(Y_turbo)

            try:
                
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                    MaternKernel(
                        nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
                    )
                )
                model = SingleTaskGP(
                    X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                
                # Do the fitting and acquisition function optimization inside the Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    # Fit the model
                    fit_gpytorch_mll(mll)

                    # Create a batch
                    X_next = generate_batch(
                        state=state,
                        model=model,
                        X=X_turbo,
                        Y=train_Y,
                        batch_size=state.batch_size,
                        n_candidates=N_CANDIDATES,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                        acqf=acq_turbo,
                        device=device,
                        dtype=dtype,
                        minimize=minimize,
                        inequality_constraints=inequality_constraints,
                        bounds=bounds,
                    )
            except Exception as e:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.error(f"Error in Turbo batch {count_batch}: {e}")
                logger.error(f"There was an error in the Turbo optimization process, we are stopping the optimization process early.")
                return

            # Evaluate the batch
            X_next_un = unnormalize(X_next, bounds=bounds)


        # load all data into ax
        # create generation strategy using the second model
        gs = self.create_generation_strategy(skip_sobol=True,skip_center=True) # we skip the sobol and center model as we already have the data and we want a trainable model

        # create ax client
        if self.ax_client is None:
            self.ax_client = Client()
        
        # Configure the experiment
        self.ax_client.configure_experiment(
            name=self.name,
            parameters=parameters_space,
            parameter_constraints=parameter_constraints,
            )
        
        self.ax_client.configure_optimization(objective=objective)

        if len(self.all_tracking_metrics) != 0:
            self.ax_client.configure_metrics([IMetric(name=m) for m in self.all_tracking_metrics])

        self.ax_client.set_generation_strategy(generation_strategy=gs)

        # add all data to ax
        X_turbo_un = unnormalize(X_turbo, bounds=bounds)
        for i in range(len(X_turbo_un)):
            dic = {}
            for j in range(len(X_turbo_un[i])):
                # check the parameter_type of the parameter_space
                if parameters_space[j].parameter_type == 'int':
                    dic[free_pnames[j]] = int(X_turbo_un[i][j].item())
                else:
                    dic[free_pnames[j]] = X_turbo_un[i][j].item()
            # add fixed params to dic
            # for p in self.params:
            #     if p.type == 'fixed':
            #         dic[p.name] = p.value
            trial_index = self.ax_client.attach_trial(parameters=dic)
            # print(trials)
            # trial_index = tr
            # add all_metrics and tracking_metrics to ax
            raw_data = {}
            for j in range(len(self.all_metrics)):
                raw_data[self.all_metrics[j]] = Y_turbo_all[i][j].item()
            if Y_tracking is not None:
                for j in range(len(self.all_tracking_metrics)):
                    raw_data[self.all_tracking_metrics[j]] = Y_tracking[i][j].item()
            self.ax_client.complete_trial(trial_index, raw_data=raw_data)

        # train the model
        # self.ax_client.get_next_trials(1) # This will train the model
        # dummy_pred = self.ax_client.predict([dic]) # This will train the model

        if self.suggest_only:
            for i in range(len(X_next_un)):
                dic = {}
                for j in range(len(X_next_un[i])):
                    if parameters_space[j].parameter_type == 'int':
                        dic[free_pnames[j]] = int(X_next_un[i][j].item())
                    else:
                        dic[free_pnames[j]] = X_next_un[i][j].item()
                # add fixed params to dic
                for p in self.params:
                    if p.type == 'fixed':
                        dic[p.name] = p.value
                trial_index = self.ax_client.attach_trial(parameters=dic)
            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info('Suggesting %d trials without running the agents', len(X_next_un))
            # put the new turbo state into kwargs_turbo
            kwargs_turbo_next_step = {}
            for key, value in state.__dict__.items():
                kwargs_turbo_next_step[key] = value
            return kwargs_turbo_next_step
        else:
            if verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                if state.restart_triggered:
                    logger.info('Turbo converged after %d batches with %d trials', count_batch-1, (count_batch-1)*state.batch_size)
                else:
                    logger.info('Turbo is terminated.')
        # if verbose_logging:
        #     logging_level = 20
        #     logger.setLevel(logging_level)
        #     logger.info('Finished Turbo')

   
######### Turbo specific functions ##############################################################
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    acceleration: float = 1.6 # Note: VLC added this to control how quickly the length scale increases/decreases. Was 2 in the original paper, now 1.6 (~golden ratio)
    restart_triggered: bool = False

    def __post_init__(self):
        # The original paper uses 4.0 / batch_size, but we use a more robust value
        # based on the dimension and batch size
        self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
            )
        
    def __init__(self, dim, batch_size, best_value, **kwargs):
        self.dim = dim
        self.batch_size = batch_size
        self.best_value = best_value
        self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
            )
        for key, value in kwargs.items():
            setattr(self, key, value)
        


def get_initial_points(dim, n_pts, seed=None, device=None, dtype=None, inequality_constraints=None,bounds=None):
    """ Generate initial points using Sobol sequence.

    Parameters
    ----------
    dim : int
        Number of dimensions
    n_pts : int
        Number of points to generate
    seed : int, optional
        Random seed, by default None
    device : torch.device, optional
        Device to use for the generated points, by default None
    dtype : torch.dtype, optional
        Data type of the generated points, by default None
    inequality_constraints : list of tuples, optional
        A list of tuples (indices, coefficients, rhs) specifying inequality constraints, by default None

    Returns
    -------
    torch.Tensor
        Generated points in the range [0, 1]^d
    """    
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    
    if inequality_constraints is None:
        return sobol.draw(n_pts).to(dtype=dtype, device=device)

    X_init = torch.empty(n_pts, dim, dtype=dtype, device=device)
    n_found = 0
    n_total_draws = 0
    max_draws = n_pts * 10000  # safety break
    
    while n_found < n_pts and n_total_draws < max_draws:
        # draw a batch of points
        n_to_draw = (n_pts - n_found) * 5 # draw more points to increase efficiency
        X_cand = sobol.draw(n_to_draw).to(dtype=dtype, device=device)

        # denormalize the candidates to check the constraints
        X_cand = unnormalize(X_cand, bounds=bounds)
        n_total_draws += n_to_draw

        # filter out candidates that violate constraints
        constraint_mask = torch.ones(n_to_draw, dtype=torch.bool, device=device)
        
        for indices, coeffs, rhs in inequality_constraints:
            # print(coeffs,rhs)
            # print(X_cand[:, indices] )
            constraint_mask &= (X_cand[:, indices] @ coeffs <= rhs)
        # print(constraint_mask)
        X_valid = X_cand[constraint_mask]
        
        n_can_add = min(n_pts - n_found, X_valid.shape[0])
        if n_can_add > 0:
            X_init[n_found : n_found + n_can_add] = normalize(X_valid[:n_can_add], bounds=bounds) # re-normalize the points
            n_found += n_can_add

    if n_found < n_pts:
        print(n_found)
        raise RuntimeError(
            f"Could not find {n_pts} initial points satisfying the constraints after drawing {n_total_draws} points. "
            "The constraints might be too strict or the parameter space too small."
        )

    return X_init

def update_state(state, Y_next):
    """ Update the state of the optimization process based on the new observations.
       For TURBO optimization only.
       The state is updated based on the success or failure of the new observations.

    Parameters
    ----------
    state : TurboState
        Current state of the optimization process
    Y_next : torch.Tensor
        New observations

    Returns
    -------
    TurboState
        Updated state of the optimization process

    """    
    # For maximization, we want the maximum value
    current_best = max(Y_next).item()
    is_success = current_best > state.best_value + 1e-3 * math.fabs(state.best_value)
    state.best_value = max(state.best_value, current_best)
    acceleration = state.acceleration
    
    if is_success:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(acceleration * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= acceleration
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def generate_batch(state, model, X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    device=None,
    dtype=None,
    minimize=False,
    inequality_constraints=None,
    bounds=None,
):
    """ Generate a batch of points using the TURBO algorithm.
    The batch is generated using either Thompson sampling or Expected Improvement.

    Parameters
    ----------
    state : TurboState
        Current state of the optimization process
    model : GPyTorchModel
        GPyTorch model for the function
    X : torch.Tensor
        Evaluated points on the domain [0, 1]^d
    Y : torch.Tensor
        Function values
    batch_size : int
        Number of points to generate
    n_candidates : int, optional
        Number of candidates for Thompson sampling, by default None
    num_restarts : int, optional
        Number of restarts for the optimization, by default 10
    raw_samples : int, optional
        Number of raw samples for the optimization, by default 512
    acqf : str, optional
        Acquisition function to use can be "ts" or "ei", by default "ts"
    device : torch.device, optional
        Device to use for the generated points, by default None
    dtype : torch.dtype, optional
        Data type of the generated points, by default None
    minimize : bool, optional
        Whether to minimize or maximize the function, by default False
    inequality_constraints : list of tuples, optional
        A list of tuples (indices, coefficients, rhs) specifying inequality constraints, by default None

    Returns
    -------
    torch.Tensor
        Generated points in the range [0, 1]^d
    
    Raises
    ------
    AssertionError
        If the acquisition function is not "ts" or "ei"
    ValueError
        If the acquisition function is not "ts" or "ei"
    """    
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
        
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        # Thompson Sampling
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True, )#seed=np.random.randint(10000))
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert
        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        if inequality_constraints is not None:
            # filter out candidates that violate constraints
            constraint_mask = torch.ones(n_candidates, dtype=torch.bool, device=device)
            X_cand = unnormalize(X_cand, bounds=bounds)
            for indices, coeffs, rhs in inequality_constraints:
                constraint_mask &= (X_cand[:, indices] @ coeffs <= rhs)
            X_cand = X_cand[constraint_mask]
            if X_cand.shape[0] < batch_size:
                logger.warning(
                    f"Reduced candidate size from {constraint_mask.shape[0]} to {X_cand.shape[0]} due to constraints. "
                    f"This may lead to suboptimal results. Consider increasing n_candidates."
                )
                if X_cand.shape[0] == 0:
                    raise RuntimeError(
                        "No candidates left after applying constraints. "
                        "Your trust region might be too small or your constraints too strict."
                    )
            X_cand = normalize(X_cand, bounds=bounds)
        # Sample from the posterior
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            
        with torch.no_grad():
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        # Expected Improvement
        if minimize:
            best_f = Y.min().item()
        else:
            best_f = Y.max().item()
        
        # Use qLogExpectedImprovement for better numerical stability
        acq_func = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
        )#
        
        X_next, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
            inequality_constraints=inequality_constraints,
        )
    else:
        raise ValueError(f"Unknown acquisition function type: {acqf}")

    return X_next

def _parse_inequality_constraints(constraints, param_names, device=None, dtype=None):
    """Parse inequality constraints from strings to botorch format."""
    inequality_constraints = []
    param_map = {name: i for i, name in enumerate(param_names)}
    
    for c in constraints:
        if "<=" in c:
            parts = c.split("<=")
            op = "<="
        elif ">=" in c:
            parts = c.split(">=")
            op = ">="
        else:
            raise ValueError(f"Invalid constraint string: {c}")

        lhs, rhs_str = parts[0].strip(), parts[1].strip()
        rhs = float(rhs_str)

        # very simple parser for linear constraints
        # handles "p1", "-p1", "c * p1", "p1 + p2", "p1 - c * p2", etc.
        pattern = r"([+\-]?)\s*([\d\.]*)\s*\*?\s*(\w+)"
        terms = re.findall(pattern, lhs)
        
        indices = []
        coeffs = []

        for sign, coeff_str, name in terms:
            if name not in param_map:
                raise ValueError(f"Unknown parameter '{name}' in constraint '{c}'")
            
            indices.append(param_map[name])
            
            if coeff_str == "":
                coeff = 1.0
            else:
                coeff = float(coeff_str)
            
            if sign == "-":
                coeff *= -1.0
            
            coeffs.append(coeff)

        indices = torch.tensor(indices, device=device, dtype=torch.long)
        coeffs = torch.tensor(coeffs, device=device, dtype=dtype)

        if op == ">=":
            # Convert >= to <=
            coeffs *= -1
            rhs *= -1

        inequality_constraints.append((indices, coeffs, rhs))
        
    return inequality_constraints