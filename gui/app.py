"""GUI for the process-correlation + condition-search part of the combinatorial-
sputtering pipeline (optimpv/ml/, scripts/optimize_process_conditions.py).

Run with:
    streamlit run gui/app.py
or use the launcher: ./run_gui.sh (Linux/Mac) / run_gui.bat (Windows)

Scope: this GUI intentionally does NOT call SIMsalabim/DDfits (Stage 1 drift-
diffusion fitting) -- it starts from a table you already have (process conditions +
performance, optionally including Stage 1 fitted physical parameters if you produced
them separately with scripts/fit_combinatorial_devices.py) and covers:

  Stage 2 -- Process Correlation: which process knob controls which physical
  parameter / performance metric, and how (feature importance + partial dependence).
  Stage 3 -- Condition Search: Bayesian-optimization search over process conditions
  using the Stage 2 model as a fast surrogate, for a ranked shortlist of conditions
  to try next.

This is a thin UI over optimpv/ml/process_correlation.py and
scripts/optimize_process_conditions.py's run_search() -- it does not duplicate any
fitting/ML/search logic.
"""
######### Package Imports #########################################################################

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from optimpv.ml.process_correlation import ProcessCorrelationModel

st.set_page_config(page_title="optimPV -- Process Correlation & Condition Search", layout="wide")

######### Helpers #####################################################################


def _check_import(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


######### Sidebar: environment status #####################################################################

with st.sidebar:
    st.title("optimPV")
    st.caption("Process correlation & condition search")
    st.markdown("---")
    st.subheader("Environment")
    has_ax = _check_import('ax')
    has_sklearn = _check_import('sklearn')
    st.write("scikit-learn (Stage 2/3 ML): " + ("available" if has_sklearn else "MISSING"))
    st.write("Ax/BoTorch (Stage 3 search): " + ("available" if has_ax else "MISSING"))
    if not has_ax:
        st.info("Install `ax-platform`/`torch` to run the Stage 3 search.")
    st.markdown("---")
    st.caption("Drift-diffusion (SIMsalabim) device fitting is intentionally not part of this GUI -- "
               "run scripts/fit_combinatorial_devices.py separately if you want physically-resolved "
               "layer parameters, or just bring a table with process conditions + measured performance.")
    st.caption("See docs/combinatorial_TCO_ML_workflow.md for the full pipeline write-up.")

######### Main tabs #####################################################################

tab_home, tab2, tab3 = st.tabs(["Overview", "Stage 2: Process Correlation", "Stage 3: Condition Search"])

with tab_home:
    st.header("Process correlation -> ML condition search")
    st.markdown(
        """
Bring a table with **one row per combinatorial device**: process-condition columns
(RF power, pressure, O2:Ar ratio, thickness, ...) plus performance and/or physical
parameter columns (PCE, Voc, FF, Jsc, and optionally drift-diffusion fitted
parameters like mobility or trap density if you ran those separately).

1. **Stage 2 -- Process Correlation**: fits a Random Forest/GBR/GP regressor from
   process conditions to your chosen target column(s), and shows feature importance
   (which knob controls what) and partial dependence (the shape of that relationship).
2. **Stage 3 -- Condition Search**: wraps the Stage 2 model as a fast surrogate and
   runs a Bayesian-optimization search over your process-condition ranges for the
   conditions predicted to maximize (or minimize) your target(s) -- a ranked
   shortlist of conditions to actually try next, not a guaranteed optimum.
        """
    )
    st.info("Physically understanding *why* a condition performs the way it does (fitted mobility, "
           "doping, trap density of your sputtered layer) requires the separate drift-diffusion "
           "fitting step -- run `python scripts/fit_combinatorial_devices.py ...` outside this app "
           "and bring its output CSV in here as your starting table.")

######### Stage 2 #####################################################################

with tab2:
    st.header("Stage 2: correlate process conditions with performance")

    f2 = st.file_uploader("Device table (CSV) -- process conditions + performance (+ optional fitted physics)",
                           type=["csv"], key="s2_upload")
    df2 = pd.read_csv(f2) if f2 is not None else None

    if df2 is not None:
        st.dataframe(df2, use_container_width=True, height=200)
        numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            feature_cols = st.multiselect("Process-condition columns (features)", numeric_cols, key="s2_features")
        with c2:
            target_cols = st.multiselect("Target column(s)", [c for c in numeric_cols if c not in feature_cols], key="s2_targets")
        with c3:
            model_kind = st.selectbox("Model", ["rf", "gbr", "gp"], key="s2_model")

        if st.button("Fit correlation model", type="primary", disabled=(not feature_cols or not target_cols or not has_sklearn)):
            model = ProcessCorrelationModel(model=model_kind).fit(df2.dropna(subset=feature_cols + target_cols), feature_cols, target_cols)
            st.session_state['stage2_model'] = model
            st.session_state['stage2_df'] = df2
            st.session_state['stage2_feature_cols'] = feature_cols
            st.session_state['stage2_target_cols'] = target_cols

            cv_r2 = model.cross_val_r2()
            st.subheader("Cross-validated R^2 (sanity-check before trusting importances/search)")
            r2_cols = st.columns(len(cv_r2))
            for col, (t, r2) in zip(r2_cols, cv_r2.items()):
                col.metric(t, f"{r2:.2f}")
                if r2 < 0.3:
                    col.caption(":warning: low R^2 -- collect more devices")

            st.subheader("Feature importance (which process knob controls what)")
            fig = model.plot_feature_importance()
            st.pyplot(fig.figure)

    if 'stage2_model' in st.session_state:
        st.subheader("Partial dependence")
        model = st.session_state['stage2_model']
        c1, c2 = st.columns(2)
        with c1:
            pd_feature = st.selectbox("Feature", st.session_state['stage2_feature_cols'], key="s2_pd_feature")
        with c2:
            pd_target = st.selectbox("Target", st.session_state['stage2_target_cols'], key="s2_pd_target")
        fig = model.plot_partial_dependence(pd_feature, pd_target)
        st.pyplot(fig.figure)

        st.caption("Reuse this table directly in the Stage 3 tab, or upload a fresh one there.")

######### Stage 3 #####################################################################

with tab3:
    st.header("Stage 3: search process conditions for the best predicted performance")

    if not has_ax:
        st.warning("ax-platform/torch are not installed in this environment -- the search below will fail until they are.")

    source3 = st.radio("Data source", ["Use table from Stage 2 tab", "Upload a CSV"], key="s3_source")
    if source3 == "Use table from Stage 2 tab":
        df3 = st.session_state.get('stage2_df')
        if df3 is None:
            st.info("No table loaded in Stage 2 yet -- fit a correlation model there first, or upload a CSV here instead.")
    else:
        f3 = st.file_uploader("Device table (CSV)", type=["csv"], key="s3_upload")
        df3 = pd.read_csv(f3) if f3 is not None else None

    if df3 is not None:
        numeric_cols3 = [c for c in df3.columns if pd.api.types.is_numeric_dtype(df3[c])]
        feature_cols3 = st.multiselect("Process-condition columns to search over", numeric_cols3, key="s3_features")

        bounds = {}
        if feature_cols3:
            st.caption("Search bounds (defaults from the data's observed range -- widen with caution, "
                       "the surrogate extrapolates poorly outside the training range)")
            for col in feature_cols3:
                lo_default, hi_default = float(df3[col].min()), float(df3[col].max())
                c1, c2 = st.columns(2)
                lo = c1.number_input(f"{col} min", value=lo_default, key=f"s3_lo_{col}")
                hi = c2.number_input(f"{col} max", value=hi_default, key=f"s3_hi_{col}")
                bounds[col] = (lo, hi)

        target_cols3 = st.multiselect("Target(s) to optimize", [c for c in numeric_cols3 if c not in feature_cols3], key="s3_targets")
        minimize_flags = []
        if target_cols3:
            st.caption("Direction per target")
            for t in target_cols3:
                minimize_flags.append(st.checkbox(f"minimize {t} (unchecked = maximize)", value=False, key=f"s3_min_{t}"))

        c1, c2, c3 = st.columns(3)
        with c1:
            model_kind3 = st.selectbox("Surrogate model", ["rf", "gbr", "gp"], key="s3_model")
        with c2:
            n_suggestions = st.slider("Top suggestions to return", 1, 20, 8, key="s3_n")
        with c3:
            run_disabled = not (feature_cols3 and target_cols3 and has_ax)

        if st.button("Run search", type="primary", disabled=run_disabled):
            from optimize_process_conditions import run_search  # noqa: E402

            status = st.empty()
            log_lines = []

            def _cb(msg):
                log_lines.append(msg)
                status.write("\n".join(log_lines))

            try:
                top, cv_r2, _ = run_search(df3.dropna(subset=feature_cols3 + target_cols3), feature_cols3, bounds,
                                            target_cols3, minimize=minimize_flags, model=model_kind3,
                                            n_suggestions=n_suggestions, progress_callback=_cb)
                st.subheader("Cross-validated R^2 (surrogate reliability)")
                st.write({t: round(r2, 3) for t, r2 in cv_r2.items()})
                for t, r2 in cv_r2.items():
                    if r2 < 0.3:
                        st.warning(f"Low R^2 for {t} ({r2:.2f}) -- treat the suggestions below as exploratory, "
                                   f"not confident predictions.")

                st.subheader("Suggested next conditions")
                st.dataframe(top, use_container_width=True)
                st.download_button("Download suggestions (CSV)", top.to_csv(index=False).encode(),
                                    file_name="suggested_next_conditions.csv")
            except Exception as e:
                st.error(f"Search failed: {e}")
