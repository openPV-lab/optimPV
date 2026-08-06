# Combinatorial-sputtering device stacks

Two `optimpv` DDfits-ready device stacks, matching the three architectures in use:

- **`CombinatorialTCO_ETL_HTL/`** -- n-i-p architecture, covers:
  - Stack 1: `glass / ITO / SnO2 / perovskite / Spiro-OMeTAD / Au` -> use `simulation_setup_niptco_ITO.txt`
  - Stack 2: `glass / FTO / SnO2 / perovskite / passivation / Spiro-OMeTAD / Au` -> use `simulation_setup_niptco_FTO_passivated.txt`
  - Both point at the same three layer files (`SnO2.txt` = l1/ETL, `Perovskite.txt` = l2, `SpiroOMeTAD.txt` = l3/HTL). They differ only in `nkTCO` (ITO vs. FTO placeholder) and are meant to differ in `l2.N_t_int` (perovskite/HTL interface trap density) to represent the passivation treatment in stack 2 -- treat `l2.N_t_int` as a `FitParam` you optimize/fit per device rather than two hard-coded values.

- **`CombinatorialSAM_ETL/`** -- p-i-n (inverted) architecture, covers:
  - Stack 3: `glass / FTO / SAMs / perovskite / passivation / C60 / BCP / Ag` -> use `simulation_setup_pinsam_FTO.txt`
  - Layer files: `Perovskite.txt` = l1 (absorber, SAM/FTO side), `C60.txt` = l2 (ETL, reused verbatim from `fakePerovskite/C60.txt`), `BCP.txt` = l3 (hole-blocking buffer).
  - **The SAM is not modeled as its own bulk layer.** At <2 nm it is a molecular monolayer, not a transport medium with a meaningful mobility/DOS -- the literature convention (and what SIMsalabim supports directly) is to represent it as a contact work-function / injection-barrier modifier: `W_L` in the simulation setup plus the composite `offset_W_L.E_v` `FitParam` (see `SIMsalabimAgent.py`'s `offset_W_*` handling). If you want an explicit ultrathin SAM layer instead, add it the same way `SnO2.txt`/`SpiroOMeTAD.txt` are wired into the n-i-p stack, at your own numerical-stability risk (sub-nm layers can be hard for the mesh).
  - Passivation (perovskite/C60 interface) is represented the same way as stack 2: `l1.N_t_int` in `Perovskite.txt`.

## Provenance of the starting parameter values

All numeric values in the new layer files (`SnO2.txt`, `SpiroOMeTAD.txt`, `BCP.txt`, `Perovskite.txt`) are **typical literature starting points** for these very commonly studied PV materials (SnO2/Spiro-OMeTAD/BCP/triple-cation perovskite), not measurements of your specific sputtered films. They exist so Stage 1 (DDfits) has a reasonable prior to start optimizing from -- the entire point of the fitting loop is that `axBOtorchOptimizer`/`scipyOpti` calibrates the process-sensitive parameters (mobility, doping, bulk/interface trap density, thickness) against each of your actual combinatorial-sputtering devices' measured J-V. Do not treat these numbers as ground truth.

Parameters flagged in-file as "process-sensitive" or "KEY process-sensitive parameter" (mainly `l1.mu_n`, `l1.N_D`, `l1.N_t_bulk`, `l1.N_t_int` for the SnO2 ETL; `L`/thickness on any layer) are the ones most likely to actually respond to your RF power / pressure / O2:Ar / deposition-time knobs -- those are the natural `FitParam(type='range', ...)` candidates in the Stage 1 fitting script, with everything else left `type='fixed'` unless a given fit is poorly constrained.

## Missing optical (n,k) data

No measured n,k files exist in this repo for FTO or SnO2, so both use `nk_ITO.txt` as a placeholder (flagged inline as `PLACEHOLDER` in each file). This is harmless for the electrical-only J-V fitting workflow used here (`genProfile = none`, matching `Notebooks/Drift_Diffusion/JV_realPerovskite.ipynb`'s approach of calibrating `G_ehp` directly from measured Jsc instead of a full optical generation profile). If you later want full optical/transfer-matrix modeling (`genProfile = calc`), supply real FTO and SnO2 n,k data first.
