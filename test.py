from optimpv import *
from optimpv.axBOtorch.axUtils import *
import pandas as pd
mun = FitParam(name = 'l2.mu_n', value = 3e-4, bounds = [1e-4,1e-3], values = None, start_value = None, log_scale = True, value_type = 'float', fscale = None, rescale = False, stepsize = None, display_name=r'$\mu_n$', unit='m$^2$ V$^{-1}$s$^{-1}$', axis_type = 'log', std = 0,encoding = None)

mup = FitParam(name = 'l2.mu_p', value = 6e-4, bounds = [1e-4,1e-3], values = None, start_value = None, log_scale = True, value_type = 'float', fscale = None, rescale = False, stepsize = None, display_name=r'$\mu_p$', unit='m$^2$ V$^{-1}$s$^{-1}$', axis_type = 'log', std = 0,encoding = None)

params = [mun, mup]

import pySIMsalabim as sim
from pySIMsalabim.experiments.JV_steady_state import *

session_path = os.path.join('/home/lecorre/Desktop/pySIMsalabim/', 'SIMsalabim','SimSS')
simss_device_parameters = os.path.join(session_path, 'simulation_setup.txt')

# Set the JV parameters
Gfracs = [0.1,0.5,1.0]  # Fractions of the generation rate to simulate
UUID = str(uuid.uuid4())

# Run the JV simulation
ret, mess = run_SS_JV(simss_device_parameters, session_path, JV_file_name='JV.dat', varFile='Var.dat', G_fracs=Gfracs, parallel=False, max_jobs=3, UUID=UUID, cmd_pars=[{'par': 'l2.L', 'val': '400e-9'}])

# Save data for fitting
X, y = [], []
for Gfrac in Gfracs:
    data = pd.read_csv(os.path.join(session_path, f'JV_Gfrac_{Gfrac}_{UUID}.dat'), sep=r'\s+')  # Load the data
    Vext = np.asarray(data['Vext'].values)
    Jext = np.asarray(data['Jext'].values)
    G = np.ones_like(Vext) * Gfrac

    if len(X) == 0:
        X = np.vstack((Vext, G)).T
        y = Jext
    else:
        X = np.vstack((X, np.vstack((Vext, G)).T))
        y = np.hstack((y, Jext))

print(X.shape)
print(y.shape)

def DD_fit(params, X, y, Gfracs, max_jobs=1, **kwargs):
    # Check if cmd_pars is in kwargs
    if 'cmd_pars' in kwargs:
        cmd_pars = kwargs['cmd_pars']
    else:
        cmd_pars = []

    for param in params:
        cmd_pars.append({'par': param.name, 'val': str(param.value)})

    # Generate random UUID
    UUID = str(uuid.uuid4())

    # Update cmd_pars and UUID in kwargs
    kwargs['cmd_pars'] = cmd_pars
    kwargs['UUID'] = UUID

    # Run the JV simulation
    ret, mess = run_SS_JV(simss_device_parameters, session_path, JV_file_name='JV.dat', varFile='Var.dat', G_fracs=Gfracs, parallel=False, max_jobs=max_jobs, **kwargs)

    # Read the data
    yfit = []
    for Gfrac in Gfracs:
        data = pd.read_csv(os.path.join(session_path, f'JV_Gfrac_{Gfrac}_{UUID}.dat'), sep=r'\s+')
        Vext = np.asarray(data['Vext'].values)
        Jext = np.asarray(data['Jext'].values)

        # Get X[:,1] == Gfrac
        V = X[X[:, 1] == Gfrac, 0]
        J = y[X[:, 1] == Gfrac]
        if len(yfit) == 0:
            # Interpolate the data
            yfit = np.interp(V, Vext, Jext)
        else:
            yfit = np.hstack((yfit, np.interp(V, Vext, Jext)))

    # Return MSE between y and yfit
    return np.mean((y - yfit) ** 2)

DD_fit(params, X, y, Gfracs, max_jobs=1)

from optimpv.axBOtorch.axUtils import *
from optimpv.general.optimizers import *
ax_params = ConvertParamsAx(params)

def evaluate(parameters, params=params, X=X, y=y, Gfracs=Gfracs):
    px = [parameters[key] for key in parameters.keys()]
    print(px)
    params = params_w(px, params)
    return DD_fit(params, X, y, Gfracs, max_jobs=1)

# Define the optimizer

from random import randint
from time import time
from typing import Any, Dict, NamedTuple, Union

from ax.core.base_trial import TrialStatus

class MockJob(NamedTuple):
    """Dummy class to represent a job scheduled on `MockJobQueue`."""
    id: int
    parameters: Dict[str, Union[str, float, int, bool]]

class MockJobQueueClient:
    """Dummy class to represent a job queue where the Ax `Scheduler` will
    deploy trial evaluation runs during optimization.
    """
    jobs: Dict[int, MockJob] = {}

    def schedule_job_with_parameters(self, parameters: Dict[str, Union[str, float, int, bool]]) -> int:
        """Schedules an evaluation job with given parameters and returns job ID."""
        job_id = int(time() * 1e6)
        self.jobs[job_id] = MockJob(job_id, parameters)
        return job_id

    def get_job_status(self, job_id: int) -> TrialStatus:
        """Get status of the job by a given ID."""
        if randint(0, 3) > 0:
            return TrialStatus.COMPLETED
        return TrialStatus.RUNNING

    def get_outcome_value_for_completed_job(self, job_id: int) -> Dict[str, float]:
        """Get evaluation results for a given completed job."""
        job = self.jobs[job_id]
        return {"branin": evaluate(job.parameters)}

MOCK_JOB_QUEUE_CLIENT = MockJobQueueClient()

def get_mock_job_queue_client() -> MockJobQueueClient:
    """Obtain the singleton job queue instance."""
    return MOCK_JOB_QUEUE_CLIENT

from collections import defaultdict
from typing import Iterable, Set

from ax.core.base_trial import BaseTrial
from ax.core.runner import Runner
from ax.core.trial import Trial

class MockJobRunner(Runner):  # Deploys trials to external system.
    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        """Deploys a trial."""
        if not isinstance(trial, Trial):
            raise ValueError("This runner only handles `Trial`.")
        mock_job_queue = get_mock_job_queue_client()
        job_id = mock_job_queue.schedule_job_with_parameters(parameters=trial.arm.parameters)
        return {"job_id": job_id}

    def poll_trial_status(self, trials: Iterable[BaseTrial]) -> Dict[TrialStatus, Set[int]]:
        """Checks the status of any non-terminal trials."""
        status_dict = defaultdict(set)
        for trial in trials:
            mock_job_queue = get_mock_job_queue_client()
            status = mock_job_queue.get_job_status(job_id=trial.run_metadata.get("job_id"))
            status_dict[status].add(trial.index)
        return status_dict

from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.core.data import Data
from ax.utils.common.result import Ok, Err

class BraninForMockJobMetric(Metric):  # Pulls data for trial from external system.
    def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:
        """Obtains data for a given trial."""
        if not isinstance(trial, Trial):
            raise ValueError("This metric only handles `Trial`.")
        try:
            mock_job_queue = get_mock_job_queue_client()
            branin_data = mock_job_queue.get_outcome_value_for_completed_job(job_id=trial.run_metadata.get("job_id"))
            df_dict = {
                "trial_index": trial.index,
                "metric_name": "branin",
                "arm_name": trial.arm.name,
                "mean": branin_data.get("branin"),
                "sem": None,
            }
            return Ok(value=Data(df=pd.DataFrame.from_records([df_dict])))
        except Exception as e:
            return Err(MetricFetchE(message=f"Failed to fetch {self.name}", exception=e))

from ax import *

def make_branin_experiment_with_runner_and_metric() -> Experiment:
    parameters = [
        RangeParameter(
            name="l2.mu_n",
            parameter_type=ParameterType.FLOAT,
            lower=1e-4,
            upper=1e-3,
            log_scale=True,
        ),
        RangeParameter(
            name="l2.mu_p",
            parameter_type=ParameterType.FLOAT,
            lower=1e-4,
            upper=1e-3,
            log_scale=True,
        ),
    ]
    objective = Objective(metric=BraninForMockJobMetric(name="branin"), minimize=True)
    return Experiment(
        name="branin_test_experiment",
        search_space=SearchSpace(parameters=parameters),
        optimization_config=OptimizationConfig(objective=objective),
        runner=MockJobRunner(),
        is_test=False,
    )

experiment = make_branin_experiment_with_runner_and_metric()

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models

generation_strategy = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,
            max_parallelism=5,
        ),
        GenerationStep(
            model=Models.GPEI,
            num_trials=-1,  # No limit
            max_parallelism=15,
        ),
    ]
)

from ax.service.scheduler import Scheduler, SchedulerOptions

scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=generation_strategy,
    options=SchedulerOptions(max_pending_trials=15,trial_type='BatchTrial',batch_size=5),
)

import numpy as np
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render, init_notebook_plotting

# init_notebook_plotting()

def get_plot():
    best_objectives = np.array(
        [[trial.objective_mean for trial in scheduler.experiment.trials.values()]]
    )
    best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Y",
    )
    return best_objective_plot

scheduler.run_n_trials(max_trials=20)

best_objective_plot = get_plot()
# render(best_objective_plot)

# Print best parameters
best_parameters = scheduler.get_best_parameters()
print(best_parameters)

from ax.service.utils.report_utils import exp_to_df

exp_to_df(experiment)

# Clean up the output files (comment out if you want to keep the output files)
sim.clean_all_output(session_path)
sim.delete_folders('tmp', session_path)