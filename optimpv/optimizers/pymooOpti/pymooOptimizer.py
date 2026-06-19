"""PymooOptimizer module. This module contains the PymooOptimizer class, which is used to optimize a given function using pymoo's genetic algorithms."""
######### Package Imports #########################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from optimpv import *
from optimpv.general.BaseAgent import BaseAgent
from torch.multiprocessing import Pool
from logging import Logger

# PyMOO imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from multiprocessing.pool import ThreadPool
try:
    from pymoo.core.problem import StarmapParallelization
except ImportError:
    from pymoo.parallelization.starmap import StarmapParallelization
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.util.ref_dirs import get_reference_directions

from optimpv.general.logger import get_logger, _round_floats_for_logging

logger: Logger = get_logger('pymooOptimizer')
ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)

######### Problem Definition #######################################################################
class PymooProblem(ElementwiseProblem):

    def __init__(self, agents, **kwargs):
        self.agents = agents
        self.params = agents.params if hasattr(agents, 'params') else agents[0].params
        self.n_var = len([p for p in self.params if p.type != 'fixed'])
        self.all_metrics = None
        self.all_metrics = self.get_all_metrics()
        self.n_obj = len(self.all_metrics)
        self.n_threads = kwargs.get('n_threads', 1)
        
        # Handle parameter constraints
        self.parameter_constraints = kwargs.get('parameter_constraints', None)
        n_constr = 0
        self.constraint_data = []
        if self.parameter_constraints is not None:
            self.constraint_data = self.parse_constraints(self.parameter_constraints)
            n_constr = len(self.constraint_data)
        
        # Check parameters and set bounds
        self.xl, self.xu = self.create_search_space(self.params)

        # Configuration for parallel evaluation
        self.parallel_agents = kwargs.get('parallel_agents', True)
        self.max_parallelism = kwargs.get('max_parallelism', -1)
        if self.max_parallelism == -1:
            self.max_parallelism = os.cpu_count() - 1
        
        if len(self.agents) == 1:  # If there is only one agent, disable parallelism
            self.parallel_agents = False

        # initialize the thread pool and create the runner
        pool = ThreadPool(self.n_threads)
        self.elementwise_runner = StarmapParallelization(pool.starmap)
        if self.n_threads > 1:
            super().__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=n_constr, 
                           xl=self.xl, xu=self.xu, elementwise_runner=self.elementwise_runner, **kwargs)
        else:
            super().__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=n_constr, 
                           xl=self.xl, xu=self.xu, **kwargs)
        

    def get_all_metrics(self):
        """Create the objectives for the optimization process."""
        append_metrics = False
        if self.all_metrics is None:
            self.all_metrics = []
            append_metrics = True

        for agent in self.agents:
            for i in range(len(agent.all_agent_metrics)):
                self.all_metrics.append(agent.all_agent_metrics[i])

        return self.all_metrics
    
    def create_search_space(self, params):
        """
        Create a search space suitable for scipy.optimize.minimize based on parameters.
        
        Parameters
        ----------
        params : list of Fitparam() objects
            List of parameter objects
            
        Returns
        -------
        tuple
            A tuple containing (x0, bounds) where x0 is the initial parameter vector
            and bounds is a list of (lower, upper) bound tuples
        """
        x0 = []
        xl, xu = [], []
        self.param_mapping = []  # Store mapping of parameter indices to names for reconstruction
        
        for i, param in enumerate(params):
            if param.type == 'fixed':
                # Fixed parameters are not included in the optimization
                continue
                
            self.param_mapping.append(param.name)
            
            if param.value_type == 'float':
                if param.force_log:
                    # For log-scale parameters, we optimize in log space
                    x0.append(np.log10(param.value))
                    xu.append(np.log10(param.bounds[1]))
                    xl.append(np.log10(param.bounds[0]))
                else:
                    # For regular float parameters
                    scale_factor = param.fscale if hasattr(param, 'fscale') else 1.0
                    x0.append(param.value / scale_factor)
                    xu.append(param.bounds[1] / scale_factor)
                    xl.append(param.bounds[0] / scale_factor)
            
            # elif param.value_type == 'int':
            #     # For integer parameters, we optimize in integer space
            #     # but will round to integers when evaluating
            #     step_size = param.stepsize if hasattr(param, 'stepsize') else 1
            #     x0.append(param.value / step_size)
            #     xu.append(param.bounds[1] / step_size)
            #     xl.append(param.bounds[0] / step_size)
            else:
                raise ValueError(f"Unsupported parameter type: {param.value_type} for parameter {param.name}")

        return np.asarray(xl), np.asarray(xu)

    def reconstruct_params(self, x):
        """
        Reconstruct a parameter dictionary from an optimization vector.
        
        Parameters
        ----------
        x : array-like
            Parameter vector from the optimizer
            
        Returns
        -------
        dict
            Dictionary mapping parameter names to values
        """
        param_dict = {}
        x_idx = 0
        for i, param in enumerate(self.params):
            if param.type == 'fixed':
                # Include fixed parameters with their fixed values
                # param_dict[param.name] = param.value
                continue
                
            # Get the optimized value for this parameter
            if param.value_type == 'float':
                if param.force_log:
                    # Convert back from log space
                    param_dict[param.name] = 10 ** x[x_idx]
                else:
                    # Apply scaling factor
                    scale_factor = param.fscale if hasattr(param, 'fscale') else 1.0
                    param_dict[param.name] = x[x_idx] * scale_factor
            
            elif param.value_type == 'int':
                # Convert back to integer with proper step size
                step_size = param.stepsize if hasattr(param, 'stepsize') else 1
                param_dict[param.name] = int(round(x[x_idx] * step_size))
            
            x_idx += 1
            
        return param_dict
            
    def evaluate_single(self, args):
        """Evaluate a single agent on a parameter point"""
        agent_idx, agent, param_dict = args
        res = agent.run_Ax(param_dict)
        return agent_idx, res
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the objective functions for a given parameter vector"""
        # Convert parameter vector to parameter dictionary
        # param_dict = self.reconstruct_params(x)
        param_dict = {}
        x_idx = 0
        
        for i, param in enumerate(self.params):
            if param.type == 'fixed':
                # Include fixed parameters with their fixed values
                # param_dict[param.name] = param.value
                continue
            param_dict[param.name] = x[x_idx]  # Initialize with the current value
            x_idx += 1

        if not self.parallel_agents:
            # Sequential evaluation: run each agent sequentially
            results = []
            for agent in self.agents:
                agent_result = agent.run_Ax(param_dict)
                results.append(agent_result)
            
            # Merge results from all agents
            merged_results = {}
            for result in results:
                merged_results.update(result)
                
        else:
            # Parallel evaluation: run all agents in parallel
            agent_param_list = []
            for idx, agent in enumerate(self.agents):
                agent_param_list.append((idx, agent, param_dict))
            
            # Run all agents in parallel using multiprocessing
            with Pool(processes=min(len(agent_param_list), self.max_parallelism)) as pool:
                parallel_results = pool.map(self.evaluate_single, agent_param_list)
            
            # Collect and merge results
            results_dict = {}
            for agent_idx, res in parallel_results:
                results_dict[agent_idx] = res
            
            # Merge results from all agents
            merged_results = {}
            for agent_idx in sorted(results_dict.keys()):
                merged_results.update(results_dict[agent_idx])
        
        # Extract objective values in the correct order
        objectives = []
        for agent in self.agents:
            for i in range(len(agent.all_agent_metrics)):
                metric_name = agent.all_agent_metrics[i]
                if metric_name in merged_results:
                    value = merged_results[metric_name]
                    # Apply sign for minimization/maximization
                    if np.isnan(value):
                        # Handle NaN values with large penalty
                        value = 1e6
                    sign = 1 if agent.minimize[i] else -1
                    objectives.append(sign * value)
                else:
                    # Handle missing metrics with large penalty
                    sign = 1 if agent.minimize[i] else -1
                    objectives.append(sign * 1e6)
        
        # Check for NaN values and handle failed evaluations
        # if any(np.isnan(obj) for obj in objectives):
        #     # Return a large penalty value for failed evaluations

        #     objectives = [1e6] * len(self.all_metrics)
        
        out["F"] = np.array(objectives)
        
        # Evaluate constraints if they exist
        if self.n_constr > 0:
            constraints = []
            for constraint in self.constraint_data:
                indices = constraint['indices']
                coefficients = constraint['coefficients']
                rhs = constraint['rhs']
                operator = constraint['operator']
                
                # Calculate constraint value: sum_i (coef[i] * x[idx[i]])
                constraint_value = np.sum(coefficients * x[indices])
                
                # Convert BoTorch-style to pymoo-style constraints
                if operator == '<=':
                    # BoTorch: sum_i coef[i]*x[idx[i]] <= rhs
                    # Pymoo: constraint_value - rhs <= 0
                    g_val = constraint_value - rhs
                elif operator == '>=':
                    # BoTorch: sum_i coef[i]*x[idx[i]] >= rhs
                    # Pymoo: rhs - constraint_value <= 0
                    g_val = rhs - constraint_value
                elif operator == '==':
                    # Equality constraint: |constraint_value - rhs| <= tolerance
                    # For pymoo, we can use constraint_value - rhs = 0
                    g_val = constraint_value - rhs
                
                constraints.append(g_val)
            
            out["G"] = np.array(constraints)


    def parse_constraints(self, constraints):
        """
        Parse parameter constraints from string format to numerical format.
        
        Parameters
        ----------
        constraints : list of str
            List of constraint strings in format like "x1 + 2*x2 <= 5"
            
        Returns
        -------
        list of dict
            List of constraint dictionaries with 'indices', 'coefficients', 'rhs', 'operator'
        """
        constraint_data = []
        param_names = [p.name for p in self.params if p.type != 'fixed']
        
        for constraint_str in constraints:
            # Parse the constraint string
            # Split by comparison operators
            if '<=' in constraint_str:
                left, right = constraint_str.split('<=')
                operator = '<='
            elif '>=' in constraint_str:
                left, right = constraint_str.split('>=')
                operator = '>='
            elif '==' in constraint_str:
                left, right = constraint_str.split('==')
                operator = '=='
            else:
                raise ValueError(f"Unsupported constraint operator in: {constraint_str}")
            
            # Parse RHS
            rhs = float(right.strip())
            
            # Parse LHS to extract coefficients and parameter indices
            indices = []
            coefficients = []
            
            # Simple parsing for linear constraints
            terms = left.replace(' ', '').replace('-', '+-').split('+')
            terms = [t for t in terms if t]  # Remove empty strings
            
            for term in terms:
                if '*' in term:
                    coef_str, param_name = term.split('*')
                    coef = float(coef_str) if coef_str not in ['', '+'] else 1.0
                    if coef_str == '-':
                        coef = -1.0
                else:
                    # Term is just a parameter name (coefficient is 1)
                    param_name = term
                    coef = 1.0
                    if param_name.startswith('-'):
                        param_name = param_name[1:]
                        coef = -1.0
                
                if param_name in param_names:
                    indices.append(param_names.index(param_name))
                    coefficients.append(coef)
                else:
                    raise ValueError(f"Parameter {param_name} not found in optimization parameters")
            
            constraint_data.append({
                'indices': np.array(indices),
                'coefficients': np.array(coefficients),
                'rhs': rhs,
                'operator': operator
            })
        
        return constraint_data

######### Optimizer Definition #######################################################################

class PymooOptimizer(BaseAgent):
    """
    Initialize the PymooOptimizer class. This class is used to optimize a given function using pymoo algorithms.
    
    Parameters
    ----------
    params : list of Fitparam() objects, optional
        List of Fitparam() objects, by default None
    agents : list of Agent() objects, optional
        List of Agent() objects see optimpv/general/BaseAgent.py for a base class definition, by default None
    algorithm : str, optional
        Optimization algorithm to use, by default 'NSGA2'
        Options: 'NSGA2', 'NSGA3', 'MOEAD', 'GA', 'DE', 'PSO'
    pop_size : int, optional
        Population size, by default 100
    n_gen : int, optional
        Number of generations, by default 100
    algorithm_kwargs : dict, optional
        Additional keyword arguments for the algorithm, by default None
    existing_data : DataFrame, optional
        existing data to use for the optimization process, by default None
    suggest_only : bool, optional
        if True, the optimization process will only suggest new points without running the agents, by default False
    name : str, optional
        Name of the optimization process, by default 'pymoo'
    **kwargs : dict
        Additional keyword arguments including:
        - parallel_agents: bool, whether to run agents in parallel
        - max_parallelism: int, maximum number of parallel processes
        - verbose_logging: bool, whether to log verbose information
        - seed: int, random seed for reproducibility
    """
    
    def __init__(self, params=None, agents=None, algorithm='NSGA2', pop_size=100, n_gen=100, 
                 algorithm_kwargs=None, existing_data = None, suggest_only = False,name='pymoo', **kwargs):
        self.params = params
        if not isinstance(agents, list):
            agents = [agents]
        self.agents = agents
        
        # Store algorithm as is (can be string or object)
        self.algorithm = algorithm
        self.algorithm_name = algorithm if isinstance(algorithm, str) else algorithm.__class__.__name__
        
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.algorithm_kwargs = algorithm_kwargs if algorithm_kwargs is not None else {}
        self.name = name
        self.kwargs = kwargs
        self.results = None
        self.all_metrics = None
        self.all_minimize = None
        self.all_metrics,self.all_minimize = self.create_metrics_list()
        self.all_evaluations = []
        self.problem = None
        self.existing_data = existing_data
        self.suggest_only = suggest_only
        # Set defaults from kwargs
        self.parallel_agents = kwargs.get('parallel_agents', True)
        self.parallel = kwargs.get('parallel', True)
        self.max_parallelism = kwargs.get('max_parallelism', os.cpu_count()-1)
        self.n_threads = kwargs.get('n_threads', int(self.max_parallelism/len(self.agents)))
        self.verbose_logging = kwargs.get('verbose_logging', True)
        self.seed = kwargs.get('seed', None)
        
        self.parameter_constraints = self.kwargs.get('parameter_constraints',None)
        self.max_attempts = self.kwargs.get('max_attempts', 10)  # Maximum attempts for constraint satisfaction
        
        if len(self.agents) == 1:
            self.parallel_agents = False

    def create_problem(self):
        """Create a pymoo problem from agents and parameters."""
        self.problem = PymooProblem(
            self.agents,
            parallel_agents=self.parallel_agents,
            max_parallelism=self.max_parallelism,
            n_threads=self.n_threads,
            parameter_constraints=self.parameter_constraints
        )
        return self.problem
    
    def create_algorithm(self):
        """Create a pymoo algorithm based on the specified algorithm name or return the provided algorithm object."""
        
        # If algorithm is already an object, return it directly
        if not isinstance(self.algorithm, str):
            return self.algorithm
        
        algorithm_map = {
            'NSGA2': NSGA2,
            'NSGA3': NSGA3,
            'MOEAD': MOEAD,
            'GA': GA,
            'DE': DE,
            'PSO': PSO,
            'CMAES': CMAES
        }
        
        # If algorithm string is not in map, return None or raise warning
        if self.algorithm not in algorithm_map:
            if self.verbose_logging:
                logger.warning(f"Warning: Algorithm '{self.algorithm}' not found in predefined algorithms. "
                      f"Available algorithms: {list(algorithm_map.keys())}")
            return None
        
        algorithm_class = algorithm_map[self.algorithm]
        
        # Set default parameters
        if self.algorithm in ['NSGA2', 'NSGA3', 'MOEAD']:
            # Multi-objective algorithms
            default_kwargs = {'pop_size': self.pop_size}
            if self.algorithm == 'NSGA3':
                # For NSGA3, we need to set the number of reference directions
                n_obj = len(self.all_metrics)
                ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=12)
                default_kwargs['ref_dirs'] = ref_dirs
            elif self.algorithm == 'MOEAD': 
                # For MOEAD, we need to set the reference directions as well
                n_obj = len(self.all_metrics)
                ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=12)
                default_kwargs = {'ref_dirs': ref_dirs, 'n_neighbors': 15, 'prob_neighbor_mating': 0.7}


        else:
            # Single-objective algorithms
            default_kwargs = {'pop_size': self.pop_size}
        
        # Update with user-provided kwargs
        default_kwargs.update(self.algorithm_kwargs)
        
        if self.seed is not None:
            default_kwargs['seed'] = self.seed
            
        return algorithm_class(**default_kwargs)
    
    

    def create_callback(self):
        """Create a callback to track optimization progress."""
        class OptimizationCallback(Callback):
            def __init__(self, optimizer):
                super().__init__()
                self.optimizer = optimizer
                
            def notify(self, algorithm):
                # Store current generation results
                if hasattr(algorithm, 'pop') and algorithm.pop is not None:
                    for ind in algorithm.pop:
                        param_dict = self.optimizer.problem.reconstruct_params(ind.X)
                        
                        # Convert objectives back to original values
                        objectives = {}
                        obj_idx = 0
                        for agent in self.optimizer.agents:
                            for i in range(len(agent.all_agent_metrics)):
                                # Convert back from minimization format
                                sign = 1 if agent.minimize[i] else -1
                                objectives[agent.all_agent_metrics[i]] = sign * ind.F[obj_idx]
                                obj_idx += 1
                        
                        self.optimizer.all_evaluations.append({
                            'params': param_dict.copy(),
                            'results': objectives.copy()
                        })
                
                if self.optimizer.verbose_logging:
                    logging_level = 20
                    logger.setLevel(logging_level)
                    f_values = algorithm.pop.get('F')
                    best_per_objective = f_values.min(axis=0)
                    objective_log = {
                        metric: float(best_per_objective[i])
                        for i, metric in enumerate(self.optimizer.all_metrics)
                    }
                    logger.info(
                        f"Generation {algorithm.n_gen}: Best objectives = {objective_log}"
)
                    # logger.info(
                    #     f"Generation {algorithm.n_gen}: Best objective = {algorithm.pop.get('F').min():.6f}"
                    # )
        
        return OptimizationCallback(self)
    
    def existing_population(self):
        """Create an existing population from the provided data."""
        
        if self.existing_data is not None:
            # Convert the existing data to a DataFrame if it is not already
            if not isinstance(self.existing_data, pd.DataFrame):
                raise ValueError("existing_data must be a pandas DataFrame")
        
        pnames = [p.name for p in self.params if p.type != 'fixed']
        # make numpy array from the pnames in the existing data
        existing_params = self.existing_data[pnames].values
        # objectives = self.existing_data[self.all_metrics].values 
        objectives = self.existing_data[self.all_metrics].to_numpy(copy=True)
        for i in range(len(self.all_minimize)):
            if not self.all_minimize[i]:
                # If the metric is to be maximized, we need to invert the values
                objectives[:, i] = -objectives[:, i]

        # Create a pymoo Population object from the existing data
        individuals = [Individual(X=existing_params[i], F=objectives[i]) for i in range(len(existing_params))]
        pop = Population.create(*individuals)

        return pop
    
    def optimize(self):
        """Run the optimization process using pymoo."""
        if self.verbose_logging:
            logging_level = 20
            logger.setLevel(logging_level)
            logger.info(
                f"Starting optimization using {self.algorithm_name} algorithm"
            )
            logger.info(
                f"Population size: {self.pop_size}, Generations: {self.n_gen}"
            )

        # Create the metrics list
        if self.all_metrics is None:
            self.all_metrics,self.all_minimize = self.create_metrics_list()

        
        # Create problem and algorithm
        problem = self.create_problem()
        algorithm = self.create_algorithm()
        callback = self.create_callback()

        if self.existing_data is not None:
            # If existing data is provided, create an initial population from it
            initial_pop = self.existing_population()
            if self.verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info(
                    f"Using existing population of size {len(initial_pop)}"
                )
            algorithm.sampling = initial_pop
            # add the initial population to self.all_evaluations
            for ind in initial_pop:
                param_dict = problem.reconstruct_params(ind.X)
                objectives = {metric: ind.F[i] for i, metric in enumerate(self.all_metrics)}
                self.all_evaluations.append({
                    'params': param_dict,
                    'results': objectives
                })

        # Run the optimization
        if self.suggest_only:
            # only suggest new points without running the agents
            if self.verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info(
                    "Suggesting new points without running agents"
                )

            # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
            algorithm.setup(problem, termination=('n_gen', 1), seed=self.seed, verbose=False)

            valid_candidates = []
            max_attempts = self.max_attempts  # Use max_attempts from kwargs
            attempt = 0
            
            while len(valid_candidates) < self.pop_size and attempt < max_attempts:
                # ask the algorithm for the next solution to be evaluated
                pop = algorithm.ask()
                
                # Check constraints for each candidate if constraints exist
                if problem.n_constr > 0:
                    for ind in pop:
                        x = ind.X
                        # Evaluate constraints
                        constraint_violations = []
                        for constraint in problem.constraint_data:
                            indices = constraint['indices']
                            coefficients = constraint['coefficients']
                            rhs = constraint['rhs']
                            operator = constraint['operator']
                            
                            # Calculate constraint value
                            constraint_value = np.sum(coefficients * x[indices])
                            
                            # Check if constraint is satisfied
                            if operator == '<=':
                                violation = constraint_value - rhs
                            elif operator == '>=':
                                violation = rhs - constraint_value
                            elif operator == '==':
                                violation = abs(constraint_value - rhs)
                            
                            constraint_violations.append(violation)
                        
                        # Only add candidate if all constraints are satisfied (violations <= tolerance)
                        tolerance = 1e-6
                        if all(violation <= tolerance for violation in constraint_violations):
                            valid_candidates.append(x)
                            
                        # Stop if we have enough valid candidates
                        if len(valid_candidates) >= self.pop_size:
                            break

                else:
                    # No constraints, all candidates are valid
                    for ind in pop:
                        valid_candidates.append(ind.X)
                        if len(valid_candidates) >= self.pop_size:
                            break
                
                attempt += 1
                
                if self.verbose_logging and problem.n_constr > 0 and attempt > 1:
                    logger.info(f"Attempt {attempt}: Found {len(valid_candidates)} valid candidates out of {self.pop_size} requested")
            
            if len(valid_candidates) == 0:
                raise ValueError("Could not generate any valid candidates that satisfy constraints")
            
            if len(valid_candidates) < self.pop_size and self.verbose_logging:
                logger.warning(f"Only found {len(valid_candidates)} valid candidates out of {self.pop_size} requested")
            
            # Take only the requested number of candidates
            valid_candidates = valid_candidates[:self.pop_size]
            
            # convert the valid candidates to a numpy array of parameters
            params_array = np.asarray(valid_candidates)
            # store the parameters to try next in a DataFrame
            params_df = pd.DataFrame(params_array, columns=[p.name for p in self.params if p.type != 'fixed'])
            # add the fixed parameters to the DataFrame
            for p in self.params:
                if p.type == 'fixed':
                    params_df[p.name] = p.value
            return params_df

        else:
            

            result = minimize(
                    problem,
                    algorithm,
                    ('n_gen', self.n_gen),
                    callback=callback,
                verbose=self.verbose_logging,
                seed=self.seed
            )
        
            self.results = result
            
            if self.verbose_logging:
                logging_level = 20
                logger.setLevel(logging_level)
                logger.info(
                    f"Optimization completed after {result.algorithm.n_gen} generations"
                )
                logger.info(
                    f"Number of function evaluations: {result.algorithm.evaluator.n_eval}"
                )

                if len(self.all_metrics) == 1:
                    logger.info(
                        f"Best objective value: {result.F.min():.6f}"
                    )
                else:
                    logger.info(
                        f"Final population size: {len(result.F)}"
                    )
                    logger.info(
                        f"Pareto front size: {len(result.F)}"
                    )

            return result
    
    def get_best_solution(self, balance_objectives=True):
        """
        Get the best solution from the optimization results.
        
        Parameters
        ----------
        balance_objectives : bool, optional
            If True and multiple objectives exist, return the solution with best balance.
            If False, return the first solution from the Pareto front.
            
        Returns
        -------
        dict
            Dictionary containing the best parameters
        """
        if self.results is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        if len(self.all_metrics) == 1 or not balance_objectives:
            # Single objective or just take first Pareto solution
            best_params = self.problem.reconstruct_params(self.results.X)
        else:
            # Multi-objective: find best balance using ranking
            if not self.all_evaluations:
                raise ValueError("No evaluations recorded. Cannot find best balance.")
            
            # Use the same logic as other optimizers
            _, best_params = self.update_params_with_best_balance(return_best_balance=True)
        
        return best_params
    
    def update_params_with_best_balance(self, return_best_balance=False):
        """Update the parameters with the best balance of all metrics.
            If multiple objectives exist, find the best balance based on ranks.

        Parameters
        ----------
        return_best_balance : bool, optional
            If True, return the index and parameters of the best balance, by default False

        Returns
        -------
        tuple
            If return_best_balance is True, return the index and parameters of the best balance.
            Otherwise, return None.

        Raises
        ------
        ValueError
            If no evaluations have been performed or no metrics are available.
        """


        if not self.all_evaluations:
            raise ValueError("No evaluations have been performed.")
        
        if len(self.all_metrics) == 0:
            raise ValueError("We need at least one metric to update the parameters")
        
        # If we have one objective, just use the best result
        if len(self.all_metrics) == 1:
            minimizes_ = []
            for agent in self.agents:
                for i in range(len(agent.minimize)):
                    minimizes_.append(agent.minimize[i])


            best_idx = np.argmin([eval_result['results'][self.all_metrics[0]] 
                                for eval_result in self.all_evaluations])
            best_params = self.all_evaluations[best_idx]['params']
            # Update parameters
            for p in self.params:
                if p.name in best_params:
                    p.value = best_params[p.name]
                    
            if return_best_balance:
                return best_idx, best_params
        else:
            # Multiple objectives: find the best balance
            metrics_values = []
            for metric in self.all_metrics:
                metric_vals = [eval_result['results'][metric] 
                             for eval_result in self.all_evaluations]
                metrics_values.append(metric_vals)
            
            # Determine minimize/maximize for each metric
            minimizes = []
            for agent in self.agents:
                for i in range(len(agent.minimize)):
                    minimizes.append(agent.minimize[i])
            
            # Calculate ranks for each metric
            ranks = []
            for i in range(len(self.all_metrics)):
                vals = np.array(metrics_values[i])
                rank = np.argsort(np.argsort(vals ))
                ranks.append(rank)
            
            # Find the best balance (lowest sum of ranks)
            sum_ranks = np.sum(ranks, axis=0)
            best_idx = np.argmin(sum_ranks)
            best_params = self.all_evaluations[best_idx]['params']
            
            # Update parameters
            for p in self.params:
                if p.name in best_params:
                    p.value = best_params[p.name]
            
            if return_best_balance:
                return best_idx, best_params
    
    def get_pareto_front(self):
        """
        Get the Pareto front solutions for multi-objective optimization.
        
        Returns
        -------
        tuple
            Tuple containing (parameters_list, objectives_array)
        """
        if self.results is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        if len(self.all_metrics) == 1:
            # Single objective: return best solution
            best_idx = np.argmin(self.results.F)
            best_params = self.problem.reconstruct_params(self.results.X[best_idx])
            return [best_params], self.results.F[best_idx:best_idx+1]
        
        # Multi-objective: return all Pareto front solutions
        pareto_params = []
        for x in self.results.X:
            params = self.problem.reconstruct_params(x)
            pareto_params.append(params)
        
        return pareto_params, self.results.F

    def plot_convergence(self, save_path=None, **kwargs):
        """
        Plot the convergence history of the optimization.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot, by default None
        **kwargs : dict
            Additional keyword arguments for the plot.

                - xscale : str, optional
                    Scale for the x-axis, by default 'linear'
                - yscale : str, optional
                    Scale for the y-axis, by default 'linear'

        """
        if not self.all_evaluations:
            raise ValueError("No evaluations recorded. Cannot plot convergence.")
        xscale = kwargs.get('xscale', 'linear')
        yscale = kwargs.get('yscale', 'linear')
        # Group evaluations by generation (assuming they are stored in order)
        n_evals_per_gen = len(self.all_evaluations) // self.n_gen
        generations = []
        best_values = []
        
        for gen in range(self.n_gen):
            start_idx = gen * n_evals_per_gen
            end_idx = (gen + 1) * n_evals_per_gen
            
            gen_evals = self.all_evaluations[start_idx:end_idx]
            
            if len(self.all_metrics) == 1:
                # Single objective: track best value
                metric = self.all_metrics[0]
                gen_values = [eval_result['results'][metric] for eval_result in gen_evals]
                best_values.append(min(gen_values))
            else:
                # Multi-objective: track hypervolume or average
                metric_values = []
                for metric in self.all_metrics:
                    metric_vals = [eval_result['results'][metric] for eval_result in gen_evals]
                    metric_values.append(np.mean(metric_vals))
                best_values.append(np.mean(metric_values))
            
            generations.append(gen)
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_values, 'b-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best Objective Value' if len(self.all_metrics) == 1 else 'Average Objective Value')
        plt.title(f'{self.algorithm_name} Convergence History')
        plt.grid(True, alpha=0.3)

        plt.xscale(xscale)
        plt.yscale(yscale)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def get_df_from_pymoo(self):
        """Convert the optimization results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the optimization results
        """
      
        resall = self.all_evaluations
        dum_dic = {}
        for key in resall[0]['params'].keys():
            dum_dic[key] = []
        for key in resall[0]['results'].keys():
            dum_dic[key] = []

        for i in range(len(resall)):
            for key in resall[i]['params'].keys():
                dum_dic[key].append(resall[i]['params'][key])
            for key in resall[i]['results'].keys():
                dum_dic[key].append(resall[i]['results'][key])

        # remove the minus from the metrics that are maximized
        for i in range(len(self.all_metrics)):
            if not self.all_minimize[i]:
                # If the metric is to be maximized, we need to invert the values
                dum_dic[self.all_metrics[i]] = [-x for x in dum_dic[self.all_metrics[i]]]
        df = pd.DataFrame(dum_dic)

        return df