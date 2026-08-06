"""Stage 1 of the combinatorial-sputtering -> device-physics -> ML pipeline.

For every device in a combinatorial-sputtering run (conditions log + raw J-V curve
files), fits the drift-diffusion (SIMsalabim) layer parameters of the sputtered
TCO/ETL/HTL layer against that device's measured J-V curve, using the appropriate
device stack (n-i-p TCO/SnO2/perovskite/Spiro-OMeTAD, or p-i-n SAM/perovskite/C60/BCP)
from Data/simsalabim_test_inputs/. The output is one row per device with process
conditions + fitted physical parameters + measured performance (Voc/Jsc/FF/PCE) --
the bridge table Stage 2 (process<->physics<->performance correlation) consumes.

This script cannot be validated end-to-end in this environment (no SIMsalabim binary
and no real data available here) -- review the FitParam bounds/fixed choices against
your actual process window before running, and treat the layer parameter starting
values as literature priors, not calibrated truth (see
Data/simsalabim_test_inputs/README_combinatorial_TCO_ETL_HTL_SAM.md).

Usage
-----
python scripts/fit_combinatorial_devices.py \\
    --conditions Data/my_run/conditions_log.csv \\
    --jv-dir Data/my_run/jv_curves \\
    --jv-filename-col jv_file \\
    --stack-col stack_type \\
    --out Data/my_run/stage1_fitted_devices.csv \\
    --method scipy
"""
######### Package Imports #########################################################################

import argparse
import copy
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimpv import FitParam
from optimpv.general.combinatorial_data import load_conditions_log, build_combinatorial_table, extract_jv_metrics
from optimpv.models.DDfits.JVAgent import JVAgent

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NIPTCO_DIR = os.path.join(REPO_ROOT, 'Data', 'simsalabim_test_inputs', 'CombinatorialTCO_ETL_HTL')
PINSAM_DIR = os.path.join(REPO_ROOT, 'Data', 'simsalabim_test_inputs', 'CombinatorialSAM_ETL')

######### Stack definitions #####################################################################
# Each stack defines: which simulation_setup file to use, the layer files to stage into the
# session directory, and the FitParam list to optimize. Only the process-sensitive parameters
# (thickness + the sputtered oxide's mobility/doping/traps) are free by default -- widen this
# list once you have enough devices per condition to constrain more parameters.

STACKS = {
    'niptco_ITO': {
        'setup': os.path.join(NIPTCO_DIR, 'simulation_setup_niptco_ITO.txt'),
        'files': ['SnO2.txt', 'Perovskite.txt', 'SpiroOMeTAD.txt', 'nk_glass.txt', 'nk_ITO.txt',
                  'nk_Au.txt', 'nk_SpiroOMeTAD.txt', 'nk_peroTripleCatMartin.txt'],
        'src_dir': NIPTCO_DIR,
    },
    'niptco_FTO_passivated': {
        'setup': os.path.join(NIPTCO_DIR, 'simulation_setup_niptco_FTO_passivated.txt'),
        'files': ['SnO2.txt', 'Perovskite.txt', 'SpiroOMeTAD.txt', 'nk_glass.txt', 'nk_ITO.txt',
                  'nk_Au.txt', 'nk_SpiroOMeTAD.txt', 'nk_peroTripleCatMartin.txt'],
        'src_dir': NIPTCO_DIR,
    },
    'pinsam_FTO': {
        'setup': os.path.join(PINSAM_DIR, 'simulation_setup_pinsam_FTO.txt'),
        'files': ['Perovskite.txt', 'C60.txt', 'BCP.txt', 'nk_glass.txt', 'nk_ITO.txt',
                  'nk_Ag.txt', 'nk_peroTripleCatMartin.txt', 'nk_BCPLiu.txt', 'nk_C60_1.txt'],
        'src_dir': PINSAM_DIR,
    },
}


def build_fit_params(stack_name):
    """Free (process-sensitive) parameters per stack. Extend/trim as your data supports."""
    if stack_name.startswith('niptco'):
        return [
            FitParam(name='l1.L', type='range', value=30e-9, bounds=[10e-9, 60e-9], log_scale=False,
                     display_name='SnO2 thickness', unit='m'),
            FitParam(name='l1.mu_n', type='range', value=1e-5, bounds=[1e-7, 1e-3], log_scale=True,
                     display_name='SnO2 mu_n', unit='m2/Vs'),
            FitParam(name='l1.N_D', type='range', value=1e23, bounds=[1e21, 1e25], log_scale=True,
                     display_name='SnO2 N_D', unit='m-3'),
            FitParam(name='l1.N_t_bulk', type='range', value=1e20, bounds=[1e18, 1e22], log_scale=True,
                     display_name='SnO2 N_t_bulk', unit='m-3'),
            FitParam(name='l2.N_t_int', type='range', value=4e12, bounds=[3e11, 5e13], log_scale=True,
                     display_name='perovskite/HTL N_t_int (passivation)', unit='m-2'),
            FitParam(name='R_series', type='range', value=1e-4, bounds=[1e-6, 1e-2], log_scale=True,
                     display_name='R_series', unit='Ohm m2'),
            FitParam(name='R_shunt', type='range', value=1e1, bounds=[1e-2, 1e3], log_scale=True,
                     display_name='R_shunt', unit='Ohm m2'),
        ]
    elif stack_name == 'pinsam_FTO':
        return [
            FitParam(name='offset_W_L.E_v', type='range', value=0.1, bounds=[-0.1, 0.5],
                     display_name='SAM injection barrier', unit='eV'),
            FitParam(name='l2.L', type='range', value=25e-9, bounds=[10e-9, 60e-9],
                     display_name='C60 thickness', unit='m'),
            FitParam(name='l2.mu_n', type='range', value=1e-6, bounds=[1e-8, 1e-4], log_scale=True,
                     display_name='C60 mu_n', unit='m2/Vs'),
            FitParam(name='l3.L', type='range', value=6e-9, bounds=[3e-9, 12e-9],
                     display_name='BCP thickness', unit='m'),
            FitParam(name='l3.mu_n', type='range', value=1e-8, bounds=[1e-10, 1e-6], log_scale=True,
                     display_name='BCP mu_n', unit='m2/Vs'),
            FitParam(name='l1.N_t_int', type='range', value=4e12, bounds=[3e11, 5e13], log_scale=True,
                     display_name='perovskite/C60 N_t_int (passivation)', unit='m-2'),
            FitParam(name='R_series', type='range', value=1e-4, bounds=[1e-6, 1e-2], log_scale=True,
                     display_name='R_series', unit='Ohm m2'),
            FitParam(name='R_shunt', type='range', value=1e1, bounds=[1e-2, 1e3], log_scale=True,
                     display_name='R_shunt', unit='Ohm m2'),
        ]
    else:
        raise ValueError(f'Unknown stack: {stack_name}')


def stage_session(session_path, stack_name):
    """Copy the stack's layer/setup files into a fresh session directory."""
    stack = STACKS[stack_name]
    os.makedirs(session_path, exist_ok=True)
    for fname in stack['files']:
        shutil.copyfile(os.path.join(stack['src_dir'], fname), os.path.join(session_path, os.path.basename(fname)))
    setup_dst = os.path.join(session_path, os.path.basename(stack['setup']))
    shutil.copyfile(stack['setup'], setup_dst)
    return setup_dst


def fit_one_device(row, stack_name, session_root, metric='nrmse', loss='linear', method='scipy'):
    """Fit one device's J-V curve, return {fit_param_name: value, ...} plus fitted metrics."""
    V, J = row['jv_voltage'], row['jv_current']
    session_path = os.path.join(session_root, str(row.name))
    simulation_setup = stage_session(session_path, stack_name)
    params = build_fit_params(stack_name)

    jv = JVAgent(params, V, J, session_path, simulation_setup, parallel=False, max_jobs=1,
                 metric=metric, loss=loss)

    if method == 'scipy':
        from optimpv.optimizers.scipyOpti.scipyOptimizer import ScipyOptimizer
        optimizer = ScipyOptimizer(params=params, agents=jv, method='L-BFGS-B')
        optimizer.optimize()
    elif method == 'bo':
        from optimpv.optimizers.axBOtorch.axBOtorchOptimizer import axBOtorchOptimizer
        num_free = len([p for p in params if p.type != 'fixed'])
        optimizer = axBOtorchOptimizer(params=params, agents=jv, models=['SOBOL', 'BOTORCH_MODULAR'],
                                        n_batches=[1, 20], batch_size=[10, 2])
        optimizer.optimize_turbo()
    else:
        raise ValueError("method must be 'scipy' or 'bo'")

    optimizer.update_params_with_best_balance()
    jv.params = optimizer.params

    fitted = {p.name: p.value for p in optimizer.params}
    yfit = jv.run(parameters={})
    fitted_metrics = extract_jv_metrics(V, yfit)
    fitted.update({f'fit_{k}': v for k, v in fitted_metrics.items()})
    return fitted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--conditions', required=True, help='Path to the conditions log CSV')
    ap.add_argument('--jv-dir', required=True, help='Directory of raw per-device J-V curve files')
    ap.add_argument('--jv-filename-col', required=True, help='Column in the conditions log giving each J-V filename')
    ap.add_argument('--stack-col', required=True,
                     help="Column in the conditions log giving the stack name (one of: %s)" % ', '.join(STACKS))
    ap.add_argument('--session-root', default='SIMsalabim_sessions', help='Where to stage per-device SIMsalabim runs')
    ap.add_argument('--out', required=True, help='Output CSV path for the merged Stage 1 table')
    ap.add_argument('--method', default='scipy', choices=['scipy', 'bo'],
                     help="Per-device fit method: 'scipy' (fast local fit, default) or 'bo' (axBOtorchOptimizer+TuRBO, slower/more robust)")
    ap.add_argument('--metric', default='nrmse')
    ap.add_argument('--loss', default='linear')
    args = ap.parse_args()

    conditions = load_conditions_log(args.conditions)
    table = build_combinatorial_table(conditions, args.jv_dir, args.jv_filename_col)

    results = []
    for idx, row in table.iterrows():
        stack_name = row[args.stack_col]
        print(f'[{idx}] fitting stack={stack_name} ...')
        try:
            fitted = fit_one_device(row, stack_name, args.session_root, metric=args.metric,
                                     loss=args.loss, method=args.method)
        except Exception as e:
            print(f'[{idx}] FAILED: {e}')
            fitted = {}
        out_row = row.drop(labels=['jv_voltage', 'jv_current']).to_dict()
        out_row.update(fitted)
        results.append(out_row)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f'Wrote {len(out_df)} fitted device(s) to {args.out}')


if __name__ == '__main__':
    main()
