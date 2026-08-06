# Combinatorial sputtering -> device physics -> ML: end-to-end workflow

This documents the 3-stage pipeline added for correlating combinatorially-sputtered
TCO/ETL/HTL conditions with device performance, understanding *why* a given condition
performs the way it does (drift-diffusion fitting), and predicting the best conditions
or layer thicknesses (ML surrogate + BO search). It targets the three device stacks:

1. `glass / ITO / SnO2 / perovskite / Spiro-OMeTAD / Au` (n-i-p)
2. `glass / FTO / SnO2 / perovskite / passivation / Spiro-OMeTAD / Au` (n-i-p, passivated)
3. `glass / FTO / SAMs / perovskite / passivation / C60 / BCP / Ag` (p-i-n, inverted)

See `Data/simsalabim_test_inputs/README_combinatorial_TCO_ETL_HTL_SAM.md` for the
layer parameter files, their provenance, and modeling choices (in particular: SAM is
represented as a contact work-function/injection-barrier tuning, not a bulk layer).

## GUI

`gui/app.py` is a Streamlit GUI covering **Stage 2 and Stage 3 only** (process
correlation + condition search) -- it deliberately does not call SIMsalabim/DDfits, so
it has no dependency on the SIMsalabim binary and works with just `scikit-learn` (plus
`ax-platform`/`torch` for the Stage 3 search). It takes as input any table with one row
per device: process-condition columns and target columns (performance, and/or Stage 1
fitted physical parameters if you produced them separately). Launch it with:

```bash
./run_gui.sh        # Linux/Mac
run_gui.bat          # Windows
# or directly:
streamlit run gui/app.py
```

Stage 1 (drift-diffusion fitting) stays a script you run separately
(`scripts/fit_combinatorial_devices.py`) -- its output CSV is exactly the kind of
table the GUI expects as input.

## Stage 1 -- physical understanding (drift-diffusion fitting per device)

`scripts/fit_combinatorial_devices.py` loops over every device in your combinatorial
run, loads its raw J-V curve, builds the matching stack's `JVAgent`, and fits the
process-sensitive physical parameters (ETL/HTL/buffer mobility, doping, bulk/interface
trap density, thickness, series/shunt resistance) against the measured curve using
either a fast local fit (`--method scipy`) or Bayesian optimization with TuRBO
(`--method bo`, slower, more robust to local minima -- prefer this for a subset of
devices that don't fit well with scipy).

Input: a conditions log CSV (one row per device, with a J-V filename column and a
stack-type column) plus a directory of raw J-V curve files. Output: one CSV row per
device with process conditions + fitted physical parameters + measured/fitted
Voc/Jsc/FF/PCE. This is the answer to "what is the TCO/ETL layer's physical role" --
e.g. plot fitted `l1.mu_n` or `l1.N_t_bulk` against `o2_ar_ratio` to see whether more
oxygen improves or degrades electron transport in your sputtered SnO2.

## Stage 2 -- correlate process conditions with physics and performance

`optimpv.ml.ProcessCorrelationModel` fits a Random Forest / Gradient Boosting / GP
regressor from process conditions to the Stage 1 fitted parameters and/or measured
performance, and exposes:

- `.cross_val_r2()` -- sanity-check the fit before trusting anything downstream
- `.feature_importance()` / `.plot_feature_importance()` -- ranked answer to "which
  process knob controls which physical parameter or performance metric"
- `.plot_partial_dependence(feature, target)` -- the shape of that relationship
  (monotonic? has an optimum? saturates?)

```python
from optimpv.ml import ProcessCorrelationModel
model = ProcessCorrelationModel(model='rf').fit(
    stage1_df,
    feature_cols=['rf_power_W', 'pressure_mTorr', 'o2_ar_ratio', 'thickness_nm'],
    target_cols=['l1.mu_n', 'l1.N_t_bulk', 'PCE'],
)
model.plot_feature_importance()
```

## Stage 3 -- predict/search for the best conditions or thickness

`scripts/optimize_process_conditions.py` trains a Stage 2 model on your Stage 1 table,
wraps it as `optimpv.ml.MLSurrogateAgent`, and runs `axBOtorchOptimizer` over the
process-condition space to find the conditions predicted to maximize PCE (or whichever
target(s) you choose). Because each "evaluation" is an instant regressor call, this
converges in seconds -- treat the output as a ranked shortlist of conditions to
actually run next (active learning: suggest -> sputter -> measure -> re-fit -> re-run
Stage 1-3), not a guaranteed optimum. It is only as trustworthy as Stage 1/2's data
coverage; the script prints cross-validated R^2 and warns if it's too low to trust the
search.

## What's genuinely new here vs. what already existed

- Device stack files (`Data/simsalabim_test_inputs/CombinatorialTCO_ETL_HTL/`,
  `CombinatorialSAM_ETL/`): new, built on the existing DDfits/SIMsalabim conventions
  (same format as `fakePerovskite`/`JVrealPerovskite`).
- `optimpv/general/combinatorial_data.py`: new data-loading/merging utility.
- `optimpv/ml/` (`ProcessCorrelationModel`, `MLSurrogateAgent`): new -- this is the
  actual "add ML beyond Bayesian optimization" piece. Everything else in `optimpv`
  (Ax/BoTorch GP surrogates, emcee/UltraNest Bayesian inference) stays untouched;
  `MLSurrogateAgent` simply lets a classical ML regressor plug into the same
  `axBOtorchOptimizer` machinery that `JVAgent`/`DiodeAgent` already use.
- `scripts/fit_combinatorial_devices.py`, `scripts/optimize_process_conditions.py`:
  new orchestration scripts tying Stage 1/2/3 together.
- `gui/app.py`, `run_gui.sh`/`run_gui.bat`: new Streamlit GUI for Stage 2/3 (see
  "GUI" section above).

## Known limitations / what still needs your input

- The layer parameter starting values are literature priors, not measurements of your
  films -- Stage 1 fitting is what calibrates them to your actual devices.
- No SIMsalabim binary or Ax/BoTorch/PyTorch were available in the environment this
  was built in, so Stage 1 (`fit_combinatorial_devices.py`) and the BO portion of
  Stage 3 (`optimize_process_conditions.py`) could not be run end-to-end here. The
  data-loading (`combinatorial_data.py`) and ML modules (`optimpv/ml/`) were verified
  against synthetic data. Run Stage 1 on a handful of real devices first and sanity
  check the fitted curves/parameters before scaling to the full combinatorial set.
- FTO and SnO2 optical (n,k) data are not shipped (ITO is reused as a placeholder,
  flagged inline) -- irrelevant for the electrical-only J-V fitting workflow used
  here (`genProfile = none`), but needed if you switch to full optical modeling.
