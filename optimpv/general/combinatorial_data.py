"""Helpers for combinatorial-sputtering datasets: loading a conditions log plus a
directory of raw per-device J-V curve files, computing solar cell figures of merit
(Voc, Jsc, FF, PCE) from a raw curve, and merging everything into one table that
Stage 1 (DDfits fitting) and Stage 2 (process-correlation ML) can consume.
"""
######### Package Imports #########################################################################

import os
import numpy as np
import pandas as pd

######### Function Definitions #####################################################################


def load_conditions_log(path, **read_csv_kwargs):
    """Load a combinatorial-sputtering conditions log (one row per device/condition).

    Parameters
    ----------
    path : str
        Path to the conditions log (CSV, or any format pandas.read_csv/read_excel can open).
    **read_csv_kwargs : dict
        Additional keyword arguments passed to pandas.read_csv (or read_excel if the
        file extension is .xls/.xlsx).

    Returns
    -------
    DataFrame
        The conditions log.
    """
    if str(path).lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(path, **read_csv_kwargs)
    return pd.read_csv(path, **read_csv_kwargs)


def load_jv_curve(path, voltage_col=None, current_col=None, sep=None, skiprows=0):
    """Load a single raw J-V curve file into (V, J) arrays.

    Tries to be forgiving about common export formats: whitespace- or comma-separated,
    with or without a header row. If voltage_col/current_col are given (as column
    names or integer positions) they are used directly; otherwise the function
    guesses the voltage column as the one that is monotonic-ish and bounded roughly
    within [-2, 2] V, and the current column as the other numeric column.

    Parameters
    ----------
    path : str
        Path to the raw J-V curve file.
    voltage_col : str or int, optional
        Column name or position holding the applied voltage [V], by default None (auto-detect).
    current_col : str or int, optional
        Column name or position holding the current density [A/m^2] (or current, if
        you rescale afterward), by default None (auto-detect).
    sep : str, optional
        Column separator, by default None (pandas infers whitespace/comma).
    skiprows : int, optional
        Number of header/metadata rows to skip before the data starts, by default 0.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Voltage array, current(-density) array, sorted by ascending voltage.
    """
    try:
        df = pd.read_csv(path, sep=sep, engine='python', skiprows=skiprows)
    except Exception:
        df = pd.read_csv(path, sep=r'\s+', engine='python', skiprows=skiprows, header=None)

    if voltage_col is None or current_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            raise ValueError(f'Could not find two numeric columns in {path}; pass voltage_col/current_col explicitly.')
        if voltage_col is None:
            # heuristic: the voltage column has the smallest range and straddles/starts near 0
            ranges = {c: (df[c].max() - df[c].min()) for c in numeric_cols}
            voltage_col = min(ranges, key=lambda c: abs(ranges[c] - 1.2))  # typical JV sweep spans ~1-1.5 V
        if current_col is None:
            current_col = [c for c in numeric_cols if c != voltage_col][0]

    V = df[voltage_col].to_numpy(dtype=float)
    J = df[current_col].to_numpy(dtype=float)
    order = np.argsort(V)
    return V[order], J[order]


def extract_jv_metrics(V, J, Pin=1000.0):
    """Compute Voc, Jsc, FF and PCE from a J-V curve.

    Sign-convention agnostic: works whether photocurrent is stored as positive or
    negative, by locating the power-generating quadrant directly.

    Parameters
    ----------
    V : array-like
        Voltage [V].
    J : array-like
        Current density [A/m^2].
    Pin : float, optional
        Incident power density [W/m^2], by default 1000.0 (1 sun, AM1.5G).

    Returns
    -------
    dict
        {'Voc': V, 'Jsc': A/m^2, 'FF': fraction, 'PCE': percent}
    """
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    order = np.argsort(V)
    V, J = V[order], J[order]

    Jsc = np.interp(0.0, V, J)
    # Voc: voltage where J crosses zero (linear interpolation between bracketing points)
    sign_change = np.where(np.diff(np.sign(J)) != 0)[0]
    if len(sign_change) == 0:
        Voc = np.nan
    else:
        i = sign_change[0]
        Voc = V[i] - J[i] * (V[i + 1] - V[i]) / (J[i + 1] - J[i])

    # Restrict the search for the max power point to between short-circuit and
    # open-circuit -- outside that window forward-bias diode current can blow up and
    # dominate |V*J| without corresponding to a physically meaningful operating point.
    if Voc == Voc:  # not NaN
        mask = (V >= 0) & (V <= Voc)
    else:
        mask = np.ones_like(V, dtype=bool)
    Pmax = np.max(np.abs(V[mask] * J[mask])) if mask.any() else np.nan
    if Voc == Voc and Jsc == Jsc and Voc * abs(Jsc) > 0:  # not NaN
        FF = Pmax / (Voc * abs(Jsc))
    else:
        FF = np.nan
    PCE = 100 * Pmax / Pin

    return {'Voc': float(Voc), 'Jsc': float(abs(Jsc)), 'FF': float(FF), 'PCE': float(PCE)}


def build_combinatorial_table(conditions_df, jv_dir, jv_filename_col, id_col=None,
                               voltage_col=None, current_col=None, Pin=1000.0):
    """Merge a conditions log with per-device raw J-V files into one table.

    For every row of `conditions_df`, loads `jv_dir/<row[jv_filename_col]>`, computes
    Voc/Jsc/FF/PCE, and appends them as new columns alongside the process conditions.

    Parameters
    ----------
    conditions_df : DataFrame
        Conditions log, e.g. from load_conditions_log().
    jv_dir : str
        Directory containing the raw per-device J-V curve files.
    jv_filename_col : str
        Column in conditions_df giving each device's J-V filename (relative to jv_dir).
    id_col : str, optional
        Column to use as a device identifier in the returned table, by default None
        (uses jv_filename_col).
    voltage_col, current_col : str or int, optional
        Passed through to load_jv_curve for each file.
    Pin : float, optional
        Incident power density [W/m^2], by default 1000.0.

    Returns
    -------
    DataFrame
        conditions_df with 'Voc', 'Jsc', 'FF', 'PCE' columns appended, plus a
        'jv_voltage'/'jv_current' object column holding the raw curve arrays (so
        Stage 1 fitting doesn't need to re-read the files from disk).
    """
    rows = []
    for _, row in conditions_df.iterrows():
        jv_path = os.path.join(jv_dir, row[jv_filename_col])
        V, J = load_jv_curve(jv_path, voltage_col=voltage_col, current_col=current_col)
        metrics = extract_jv_metrics(V, J, Pin=Pin)
        out = row.to_dict()
        out.update(metrics)
        out['jv_voltage'] = V
        out['jv_current'] = J
        rows.append(out)

    table = pd.DataFrame(rows)
    if id_col is None:
        id_col = jv_filename_col
    table = table.set_index(id_col, drop=False)
    return table
