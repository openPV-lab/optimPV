"""Stage 3 of the combinatorial-sputtering -> device-physics -> ML pipeline.

Trains a Stage 2 ProcessCorrelationModel on the Stage 1 output table (process
conditions + fitted physical parameters + measured performance, one row per device),
wraps it as an MLSurrogateAgent, and runs axBOtorchOptimizer over the *process*
parameter space (RF power, pressure, O2:Ar ratio, thickness for ETL/HTL/TCO/PVK) to
predict which conditions maximize PCE (or whichever target(s) you choose). Since each
"evaluation" is an instant regressor call rather than a real sputter-and-measure cycle,
this is cheap to run to convergence -- treat its output as a ranked shortlist of
conditions to actually run next (active learning), not a guaranteed optimum: it is
only as good as Stage 1/2's data coverage and the regressor's extrapolation.

The reusable logic lives in `run_search()` so both the CLI below and the GUI
(gui/app.py) call the same code path. Usage as a CLI:

python scripts/optimize_process_conditions.py \\
    --stage1-csv Data/my_run/stage1_fitted_devices.csv \\
    --feature-cols rf_power_W pressure_mTorr o2_ar_ratio thickness_nm \\
    --feature-bounds 50 300 5 50 0.0 0.5 10 60 \\
    --targets PCE \\
    --out Data/my_run/suggested_next_conditions.csv \\
    --n-suggestions 8
"""
######### Package Imports #########################################################################

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimpv import FitParam
from optimpv.ml import ProcessCorrelationModel, MLSurrogateAgent


def run_search(df, feature_cols, bounds, targets, minimize=None, model='rf', n_suggestions=8,
                n_batches=None, batch_size=None, progress_callback=None):
    """Fit a Stage 2 model and search process-condition space for the best predicted targets.

    Parameters
    ----------
    df : DataFrame
        Stage 1 output table (or anything with feature_cols + targets columns).
    feature_cols : list of str
        Process-condition columns to search over.
    bounds : dict
        {feature_col: (low, high)} search bounds for each feature.
    targets : list of str
        Target column(s) to optimize.
    minimize : list of bool, optional
        Per-target minimize flag, by default None (maximize every target).
    model : str, optional
        'rf', 'gbr' or 'gp', by default 'rf'.
    n_suggestions : int, optional
        How many top process-condition points to return, by default 8.
    n_batches, batch_size : list of int, optional
        Passed to axBOtorchOptimizer; sensible defaults are used if None.
    progress_callback : callable, optional
        If given, called as progress_callback(message: str) at each major step
        (useful for streaming status into a GUI).

    Returns
    -------
    (DataFrame, dict, ProcessCorrelationModel)
        (top suggested conditions + predicted metrics, cross-validated R^2 per
        target, the fitted Stage 2 model)
    """
    def log(msg):
        if progress_callback is not None:
            progress_callback(msg)

    missing = [c for c in feature_cols + targets if c not in df.columns]
    if missing:
        raise ValueError(f'Columns missing from input table: {missing}')
    df = df.dropna(subset=feature_cols + targets)
    if len(df) < 5:
        raise ValueError(f'Only {len(df)} usable row(s) after dropping missing values -- need at least a handful of devices to fit a surrogate.')

    if minimize is None:
        minimize = [False] * len(targets)

    log('Fitting Stage 2 process-correlation model...')
    corr_model = ProcessCorrelationModel(model=model).fit(df, feature_cols, targets)
    cv_r2 = corr_model.cross_val_r2()
    log('Cross-validated R^2: ' + ', '.join(f'{t}={r2:.3f}' for t, r2 in cv_r2.items()))

    params = [FitParam(name=col, type='range', value=float(df[col].mean()), bounds=list(bounds[col]),
                        display_name=col) for col in feature_cols]

    agent = MLSurrogateAgent(params, corr_model, targets=targets, minimize=minimize)

    from optimpv.optimizers.axBOtorch.axBOtorchOptimizer import axBOtorchOptimizer

    num_free = len(params)
    n_batches = n_batches or [max(10, 2 * num_free), 30]
    batch_size = batch_size or [1, 1]
    log(f'Running BO search over {num_free} process parameter(s)...')
    optimizer = axBOtorchOptimizer(params=params, agents=agent, models=['SOBOL', 'BOTORCH_MODULAR'],
                                    n_batches=n_batches, batch_size=batch_size)
    optimizer.optimize()

    data = optimizer.ax_client.summarize()
    sort_col = optimizer.all_metrics[0]
    ascending = agent.minimize[0]
    top = data.sort_values(sort_col, ascending=ascending).head(n_suggestions)
    top = top[feature_cols + optimizer.all_metrics].reset_index(drop=True)
    log(f'Done. Top {len(top)} suggested condition(s) ready.')
    return top, cv_r2, corr_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage1-csv', required=True, help='Stage 1 output CSV (from fit_combinatorial_devices.py)')
    ap.add_argument('--feature-cols', nargs='+', required=True, help='Process-condition columns, e.g. rf_power_W pressure_mTorr o2_ar_ratio thickness_nm')
    ap.add_argument('--feature-bounds', nargs='+', type=float, required=True,
                     help='Lower/upper bound pairs, in the same order as --feature-cols, e.g. lo1 hi1 lo2 hi2 ...')
    ap.add_argument('--targets', nargs='+', required=True, help='Target column(s) to optimize, e.g. PCE or PCE Voc FF')
    ap.add_argument('--minimize', nargs='+', default=None,
                     help="Per-target minimize flag ('true'/'false'); default maximize (false) for every target")
    ap.add_argument('--model', default='rf', choices=['rf', 'gbr', 'gp'])
    ap.add_argument('--n-suggestions', type=int, default=8, help='How many top process-condition points to report')
    ap.add_argument('--out', required=True, help='Output CSV path for the suggested conditions')
    args = ap.parse_args()

    if len(args.feature_bounds) != 2 * len(args.feature_cols):
        raise ValueError('--feature-bounds must give exactly 2 values (lo, hi) per --feature-cols entry')
    bounds = {col: (args.feature_bounds[2 * i], args.feature_bounds[2 * i + 1])
              for i, col in enumerate(args.feature_cols)}
    minimize = [False] * len(args.targets) if args.minimize is None else [m.lower() == 'true' for m in args.minimize]

    df = pd.read_csv(args.stage1_csv)
    top, cv_r2, _ = run_search(df, args.feature_cols, bounds, args.targets, minimize=minimize,
                                model=args.model, n_suggestions=args.n_suggestions, progress_callback=print)

    print('\nCross-validated R^2 per target (sanity-check before trusting the search above):')
    for t, r2 in cv_r2.items():
        print(f'  {t}: {r2:.3f}')
        if r2 < 0.3:
            print(f'    WARNING: low R^2 for {t} -- collect more Stage 1 devices or narrow the process window.')

    top.to_csv(args.out, index=False)
    print(f'\nWrote top {len(top)} suggested process condition(s) to {args.out}')
    print(top.to_string(index=False))


if __name__ == '__main__':
    main()
