"""Module containing classes and functions for running Ax optimization with with a scheduler that usese a multiprocessing pool to run the jobs in parallel."""

######### Package Imports #########################################################################
import os,sys,json,uuid,time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Any, Dict, NamedTuple, Union, Iterable, Set
import ax
from ax import *
from ax.core.base_trial import BaseTrial
from ax.core.base_trial import TrialStatus
from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.core.data import Data
from ax.utils.common.result import Ok, Err
from ax.core.runner import Runner
from ax.core.trial import Trial
from collections import defaultdict

######### Function Definitions ####################################################################
class MockJob(NamedTuple):
    """Dummy class to represent a job scheduled on `MockJobQueue`."""

    id: int
    parameters: Dict[str, Union[str, float, int, bool]]

    def run(self, job_id, parameters, agents = None, tmp_dir = None, parallel_agents = True):
        
        if parallel_agents:
            res_dic = {}
            results = Parallel(n_jobs=len(agents))(delayed(agents[i].run_Ax)(parameters) for i in range(len(agents)))
            for i in range(len(agents)):
                res_dic.update(results[i])
        else:
            res_dic = {}
            for i in range(len(agents)):
                res_dic.update(agents[i].run_Ax(parameters))

        # save the results in tmp folder with the job_id in json format
        if tmp_dir is not None:
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            with open(os.path.join(tmp_dir,str(job_id)+'.json'), 'w') as fp:
                json.dump(res_dic, fp)

class MockJobQueueClient:
        """Dummy class to represent a job queue where the Ax `Scheduler` will
        deploy trial evaluation runs during optimization.
        """

        jobs: Dict[str, MockJob] = {}

        def __init__(self, agents =  None, pool = None, tmp_dir = None, parallel_agents = True):
            self.agents = agents
            self.pool = pool
            self.tmp_dir = tmp_dir
            self.parallel_agents = parallel_agents

        def schedule_job_with_parameters(
            self, parameters: Dict[str, Union[str, float, int, bool]]
        ) -> int:
            """Schedules an evaluation job with given parameters and returns job ID."""
            # Code to actually schedule the job and produce an ID would go here;
            job_id = str(uuid.uuid4())
            mock = MockJob(job_id, parameters)
            # add mock run to the queue q 
            self.jobs[job_id] = MockJob(job_id, parameters)
            self.pool.apply_async(self.jobs[job_id].run, args=(job_id, parameters, self.agents, self.tmp_dir, self.parallel_agents))

            return job_id

        def get_job_status(self, job_id: str) -> TrialStatus:
            """ "Get status of the job by a given ID. For simplicity of the example,
            return an Ax `TrialStatus`.
            """
            job = self.jobs[job_id]
            # check if job_id.json exists in the tmp directory
            if os.path.exists(os.path.join(self.tmp_dir,str(job_id)+'.json')):
                #load the results
                with open(os.path.join(self.tmp_dir,str(job_id)+'.json'), 'r') as fp:
                    res_dic = json.load(fp)

                # check is nan in res_dic
                for key in res_dic.keys():
                    if np.isnan(res_dic[key]):
                        print(f'Job {job_id} failed')
                        return TrialStatus.FAILED
                    
                return TrialStatus.COMPLETED
            else:
                return TrialStatus.RUNNING

        def get_outcome_value_for_completed_job(self, job_id: int) -> Dict[str, float]:
            """Get evaluation results for a given completed job."""
            job = self.jobs[job_id]
            # In a real external system, this would retrieve real relevant outcomes and
            # not a synthetic function value.
            # check if job_id.json exists in the tmp directory
            if os.path.exists(os.path.join(self.tmp_dir,str(job_id)+'.json')):
                #load the results
                with open(os.path.join(self.tmp_dir,str(job_id)+'.json'), 'r') as fp:
                    res_dic = json.load(fp)
                # delete file
                # os.remove(os.path.join(self.tmp_dir,str(job_id)+'.json'))
                # print('WE ARE DELETING THE FILE')
                return res_dic
            else:
                raise ValueError('The job is not completed yet')



def get_mock_job_queue_client(MOCK_JOB_QUEUE_CLIENT) -> MockJobQueueClient:
        """Obtain the singleton job queue instance."""
        return MOCK_JOB_QUEUE_CLIENT


class MockJobRunner(Runner):  # Deploys trials to external system.

    def __init__(self, agents = None, pool = None, tmp_dir = None, parallel_agents = True):
        """Initializes the `MockJobRunner`.

        Parameters
        ----------
        agents : list of Agent() objects, optional
        List of Agent() objects see optimpv/general/BaseAgent.py for a base class definition, by default None
        pool : process pool, optional
            process pool object for parallel processing, by default None
        tmp_dir : str, optional
            path to the temporary directory to store the results, by default None
        parallel_agents : bool, optional
            if True the agents will be run in parallel, by default True
        """        
        self.agents = agents
        self.pool = pool
        self.tmp_dir = tmp_dir
        self.parallel_agents = parallel_agents
        self.MOCK_JOB_QUEUE_CLIENT = MockJobQueueClient(agents = self.agents, pool = self.pool, tmp_dir = self.tmp_dir, parallel_agents = self.parallel_agents)

    def _get_mock_job_queue_client(self) -> MockJobQueueClient:
        """Obtain the singleton job queue instance."""
        return self.MOCK_JOB_QUEUE_CLIENT
    
    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        if not isinstance(trial, Trial) and not isinstance(trial, BatchTrial):
            raise ValueError("This runner only handles `Trial`.")

        mock_job_queue = self._get_mock_job_queue_client()

        run_metadata = []
        if isinstance(trial, BatchTrial):
            for arm in trial.arms:
                job_id = mock_job_queue.schedule_job_with_parameters(
                    parameters=arm.parameters
                )
                # This run metadata will be attached to trial as `trial.run_metadata`
                # by the base `Scheduler`.
                arm.run_metadata = {"job_id": job_id}
        else:
            job_id = mock_job_queue.schedule_job_with_parameters(
                parameters=trial.arm.parameters
            )

        # This run metadata will be attached to trial as `trial.run_metadata`
        # by the base `Scheduler`.
        return {"job_id": job_id}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        """Checks the status of any non-terminal trials and returns their
        indices as a mapping from TrialStatus to a list of indices. Required
        for runners used with Ax ``Scheduler``.

        NOTE: Does not need to handle waiting between polling calls while trials
        are running; this function should just perform a single poll.

        Args:
            trials: Trials to poll.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that have
            the respective status at the time of the polling. This does not need to
            include trials that at the time of polling already have a terminal
            (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        status_dict = defaultdict(set)
        for trial in trials:
            if isinstance(trial, BatchTrial):
                all_status = []
                for arm in trial.arms:
                    mock_job_queue = self._get_mock_job_queue_client()
                    status = mock_job_queue.get_job_status(
                        job_id=arm.run_metadata.get("job_id")
                    )
                    all_status.append(status)
                if all(status == TrialStatus.COMPLETED for status in all_status):
                    status_dict[status].add(trial.index)
                elif all(status == TrialStatus.FAILED for status in all_status):
                    status_dict[status].add(trial.index)
                # if one arm is running the whole trial is running
                elif any(status == TrialStatus.RUNNING for status in all_status):
                    status_dict[TrialStatus.RUNNING].add(trial.index)
                # if none are running and all are either completed or failed
                elif all(status == TrialStatus.FAILED or status == TrialStatus.COMPLETED for status in all_status):
                    status_dict[TrialStatus.COMPLETED].add(trial.index)
            else:
                mock_job_queue = self._get_mock_job_queue_client()
                status = mock_job_queue.get_job_status(
                    job_id=trial.run_metadata.get("job_id")
                )
                status_dict[status].add(trial.index)

        return status_dict
    
class BraninForMockJobMetric(Metric):  # Pulls data for trial from external system.
    def __init__(self, name = None, agents = None, pool = None, tmp_dir = None, parallel_agents = True, **kwargs):
        self.agents = agents
        self.pool = pool
        self.tmp_dir = tmp_dir
        self.parallel_agents = parallel_agents
        self.MOCK_JOB_QUEUE_CLIENT = MockJobQueueClient(agents = self.agents, pool = self.pool, tmp_dir = self.tmp_dir, parallel_agents = self.parallel_agents)
        super().__init__(name=name, **kwargs)

    def _get_mock_job_queue_client(self) -> MockJobQueueClient:
        """Obtain the singleton job queue instance."""
        return self.MOCK_JOB_QUEUE_CLIENT

    def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:
        """Obtains data via fetching it from ` for a given trial."""
        if not isinstance(trial, Trial) and not isinstance(trial, BatchTrial):
            raise ValueError("This metric only handles `Trial`.")

        try:
            mock_job_queue = self._get_mock_job_queue_client()

            # Here we leverage the "job_id" metadata created by `MockJobRunner.run`.
            if isinstance(trial, BatchTrial):
                lst_df_dict = []
                for arm in trial.arms:
                    job_id = arm.run_metadata.get("job_id")
                    while not os.path.exists(os.path.join(self.tmp_dir,str(job_id)+'.json')):
                        time.sleep(.1)

                    # branin_data = mock_job_queue.get_outcome_value_for_completed_job(
                    #     job_id=trial.run_metadata.get("job_id")
                    # )
                    # arm.run_metadata.get("job_id")
                    branin_data = mock_job_queue.get_outcome_value_for_completed_job(
                        job_id=arm.run_metadata.get("job_id")
                    )

                    name_ = list(branin_data.keys())[0]
                    if np.isnan(branin_data.get(self.name)):
                        # trial.mark_arm_abandoned(arm_name=arm.name)
                        continue
                    if isinstance(branin_data.get(self.name), tuple):
                        mean_ = branin_data.get(self.name)[0]
                        sem_ = branin_data.get(self.name)[1]
                    else:
                        mean_ = branin_data.get(self.name)
                        sem_ = None
                    df_dict = {
                        "trial_index": trial.index,
                        "metric_name": self.name,
                        "arm_name": arm.name,
                        "mean": mean_,
                        "sem": sem_,
                    }
                    lst_df_dict.append(df_dict)
                return Ok(value=Data(df=pd.DataFrame.from_records(lst_df_dict)))
            else:

                job_id = trial.run_metadata.get("job_id")
                while not os.path.exists(os.path.join(self.tmp_dir,str(job_id)+'.json')):
                    time.sleep(.1)
                branin_data = mock_job_queue.get_outcome_value_for_completed_job(
                        job_id=arm.run_metadata.get("job_id")
                    )
                name_ = list(branin_data.keys())[0]
                if np.isnan(branin_data.get(self.name)):
                    trial.mark_as_failed()
                    return Ok(value=Data(df=pd.DataFrame()))
                
                if isinstance(branin_data.get(self.name), tuple):
                    mean_ = branin_data.get(self.name)[0]
                    sem_ = branin_data.get(self.name)[1]
                else:
                    mean_ = branin_data.get(self.name)
                    sem_ = None

                df_dict = {
                    "trial_index": trial.index,
                    "metric_name": self.name,
                    "arm_name": arm.name,
                    "mean": mean_,
                    "sem": sem_,
                }
                return Ok(value=Data(df=pd.DataFrame.from_records([df_dict])))
        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )


if __name__ == '__main__':
    pass