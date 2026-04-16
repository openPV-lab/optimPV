"""Lightweight tests for external Ax generation nodes."""
# example run from the root directory: `python -m pytest optimpv/tests/test_externalGenerationNode.py -q`
# or for individual tests: `python -m pytest optimpv/tests/test_externalGenerationNode.py -q -k casmopolitan||turbo||reiturbo||morbo`

######### Package Imports #########################################################################

import os
import sys

import torch

try:
    from optimpv import *
except Exception:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from optimpv import *

from optimpv.optimizers.axBOtorch.axBOtorchOptimizer import axBOtorchOptimizer


######### Test Functions ##########################################################################

def test_turbo_generation_node():
    """Run a minimal TuRBO optimization to verify the node can execute."""
    try:
        from optimpv.tests.Hartmann.HartmannAgent import HartmannAgent

        params = []
        for i in range(3):
            params.append(
                FitParam(
                    name=f"x{i}",
                    value=0.5,
                    bounds=[0.0, 1.0],
                    value_type="float",
                    type="range",
                    display_name=f"x{i}",
                    axis_type="linear",
                )
            )

        hartmann = HartmannAgent(
            params=params,
            dim=3,
            metric="obj",
            loss="linear",
            minimize=True,
            name="hartmann",
        )

        optimizer = axBOtorchOptimizer(
            params=params,
            agents=hartmann,
            models=["SOBOL", "TURBO"],
            n_batches=[1, 2],
            batch_size=[4, 1],
            model_kwargs_list=[
                {},
                {
                    "torch_device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    "torch_dtype": torch.double,
                    "acq": "ts",
                    "seed": 1,
                },
            ],
            max_parallelism=1,
            verbose_logging=False,
            name="test_turbo_generation_node",
        )

        optimizer.optimize()

        completed_trials = sum(
            trial.status.is_completed for trial in optimizer.ax_client._experiment.trials.values()
        )
        assert completed_trials == 6
    except Exception as e:
        assert False, f"TuRBO smoke test failed: {e}"


def test_reiturbo_generation_node():
    """Run a minimal REI-TuRBO optimization to verify the node can execute."""
    try:
        from optimpv.tests.Hartmann.HartmannAgent import HartmannAgent

        params = []
        for i in range(3):
            params.append(
                FitParam(
                    name=f"x{i}",
                    value=0.5,
                    bounds=[0.0, 1.0],
                    value_type="float",
                    type="range",
                    display_name=f"x{i}",
                    axis_type="linear",
                )
            )

        hartmann = HartmannAgent(
            params=params,
            dim=3,
            metric="obj",
            loss="linear",
            minimize=True,
            name="hartmann",
        )

        optimizer = axBOtorchOptimizer(
            params=params,
            agents=hartmann,
            models=["SOBOL", "REITURBO"],
            n_batches=[1, 2],
            batch_size=[4, 1],
            model_kwargs_list=[
                {},
                {
                    "torch_device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    "torch_dtype": torch.double,
                    "acq": "ts",
                    "n_trust_regions": 1,
                    "region_init_points": 4,
                    "racqf": "REI",
                    "seed": 1,
                },
            ],
            max_parallelism=1,
            verbose_logging=False,
            name="test_reiturbo_generation_node",
        )

        optimizer.optimize()

        completed_trials = sum(
            trial.status.is_completed for trial in optimizer.ax_client._experiment.trials.values()
        )
        assert completed_trials == 6
    except Exception as e:
        assert False, f"REI-TuRBO smoke test failed: {e}"


def test_morbo_generation_node():
    """Run a minimal MORBO optimization on HPA201 to verify the node can execute."""
    try:
        from optimpv.tests.HPA import HPAAgent, HPAModel

        hpa_model = HPAModel(
            problem_name="HPA201",
            n_div=4,
            level=0,
            normalized=True,
        )

        params = []
        for i in range(hpa_model.nx):
            params.append(
                FitParam(
                    name=f"x{i}",
                    value=0.5,
                    bounds=[0.0, 1.0],
                    value_type="float",
                    type="range",
                    display_name=f"x{i}",
                    axis_type="linear",
                )
            )

        hpa_agent = HPAAgent(
            params=params,
            problem_name="HPA201",
            n_div=4,
            level=0,
            normalized=True,
            metric=["f1", "f2"],
            loss="linear",
            minimize=[True, True],
            name="hpa_moo",
        )

        optimizer = axBOtorchOptimizer(
            params=params,
            agents=hpa_agent,
            models=["SOBOL", "MORBO"],
            n_batches=[1, 1],
            batch_size=[8, 2],
            model_kwargs_list=[
                {},
                {
                    "torch_device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    "torch_dtype": torch.double,
                    "reference_point": [300.0, 10.0],
                    "max_evals": 10,
                    "n_initial_points": 8,
                    "tr_hparam_overrides": {
                        "min_tr_size": 8,
                        "max_tr_size": 20,
                        "n_trust_regions": 2,
                        "raw_samples": 128,
                    },
                    "verbose": False,
                },
            ],
            max_parallelism=1,
            verbose_logging=False,
            name="test_morbo_generation_node",
        )

        optimizer.optimize()

        completed_trials = sum(
            trial.status.is_completed for trial in optimizer.ax_client._experiment.trials.values()
        )
        assert completed_trials == 10
    except Exception as e:
        assert False, f"MORBO smoke test failed: {e}"


def test_casmopolitan_generation_node():
    """Run a minimal Casmopolitan optimization to verify the node can execute."""
    try:
        from optimpv.models.TransferMatrix.TransferMatrixAgent import TransferMatrixAgent

        params = []
        params.append(
            FitParam(
                name="d_3",
                value=80e-9,
                bounds=[40e-9, 200e-9],
                log_scale=False,
                rescale=True,
                stepsize=1e-9,
                value_type="float",
                type="range",
                display_name="d_3",
                unit="m",
            )
        )
        params.append(
            FitParam(
                name="d_6",
                value=10e-9,
                bounds=[5e-9, 20e-9],
                log_scale=False,
                rescale=True,
                stepsize=1e-9,
                value_type="float",
                type="range",
                display_name="d_6",
                unit="m",
            )
        )
        params.append(
            FitParam(
                name="d_7",
                value=100e-9,
                bounds=[50e-9, 200e-9],
                log_scale=False,
                rescale=True,
                stepsize=1e-9,
                value_type="float",
                type="range",
                display_name="d_7",
                unit="m",
            )
        )
        params.append(
            FitParam(
                name="d_8",
                value=10e-9,
                bounds=[5e-9, 20e-9],
                log_scale=False,
                rescale=True,
                stepsize=1e-9,
                value_type="float",
                type="range",
                display_name="d_8",
                unit="m",
            )
        )
        params.append(
            FitParam(
                name="d_9",
                value=100e-9,
                bounds=[50e-9, 200e-9],
                log_scale=False,
                rescale=True,
                stepsize=1e-9,
                value_type="float",
                type="range",
                display_name="d_9",
                unit="m",
            )
        )
        params.append(
            FitParam(
                name="nk_3",
                value="PCE10_FOIC_1to1",
                values=["PCE10_FOIC_1to1", "P3HTPCBM_BHJ", "PM6Y6Brabec"],
                log_scale=False,
                rescale=False,
                value_type="str",
                type="choice",
                display_name="nk_3",
                unit="",
            )
        )

        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        mat_dir = os.path.join(parent_dir, "Data", "matdata")
        layers = ["SiOx", "ITO", "ZnO", "PCE10_FOIC_1to1", "MoOx", "Ag", "MoOx", "LiF", "MoOx", "LiF", "Air"]
        thicknesses = [0, 100e-9, 30e-9, 100e-9, 9e-9, 8e-9, 100e-9, 100e-9, 100e-9, 100e-9, 100e-9]

        tm_agent = TransferMatrixAgent(
            params,
            [None],
            layers=layers,
            thicknesses=thicknesses,
            lambda_min=350e-9,
            lambda_max=800e-9,
            lambda_step=1e-9,
            x_step=1e-9,
            activeLayer=3,
            spectrum=os.path.join(mat_dir, "AM15G.txt"),
            mat_dir=mat_dir,
            photopic_file=os.path.join(mat_dir, "photopic_curve.txt"),
            exp_format=["LUE"],
            metric=[None],
            loss=[None],
            threshold=[0],
            minimize=[False],
        )

        optimizer = axBOtorchOptimizer(
            params=params,
            agents=tm_agent,
            models=["SOBOL", "CASMOPOLITAN"],
            n_batches=[1, 1],
            batch_size=[4, 1],
            model_kwargs_list=[
                {"seed": 1},
                {
                    "acq": "ts",
                    "n_training_steps": 30,
                    "guided_restart": False,
                },
            ],
            max_parallelism=1,
            verbose_logging=False,
            name="test_casmopolitan_generation_node",
        )

        optimizer.optimize()

        completed_trials = sum(
            trial.status.is_completed for trial in optimizer.ax_client._experiment.trials.values()
        )
        assert completed_trials == 5
    except Exception as e:
        assert False, f"Casmopolitan smoke test failed: {e}"
