import os
import sys
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ultranest
import xarray as xr

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

try:
    from optimpv import *
    from optimpv.optimizers.axBOtorch.axUtils import *
except Exception:
    sys.path.append("../..")  # Add the project root when running the script directly.
    from optimpv import *
    from optimpv.optimizers.axBOtorch.axUtils import *


# -----------------------------------------------------------------------------
# Define parameters to be optimized
# -----------------------------------------------------------------------------
params = []  # list of parameters to be optimized

mun = FitParam(name='l2.mu_n', value=7e-8, bounds=[1e-9,1e-6], log_scale=True, value_type='float', fscale=None, rescale=False, display_name=r'$\mu_n$', unit='m$^2$ V$^{-1}$s$^{-1}$', axis_type='log', force_log=True)
params.append(mun)

mup = FitParam(name='l2.mu_p', value=5e-8, bounds=[1e-9,1e-6], log_scale=True, value_type='float', fscale=None, rescale=False, display_name=r'$\mu_p$', unit=r'm$^2$ V$^{-1}$s$^{-1}$', axis_type='log', force_log=True)
params.append(mup)

bulk_tr = FitParam(name='l2.N_t_bulk', value=1e20, bounds=[1e19,1e22], log_scale=True, value_type='float', fscale=None, rescale=False, display_name=r'$N_{T}$', unit=r'm$^{-3}$', axis_type='log', force_log=True)
params.append(bulk_tr)

preLangevin = FitParam(name='l2.preLangevin', value=1e-2, bounds=[0.005,1], log_scale=True, value_type='float', fscale=None, rescale=False, display_name=r'$\gamma_{pre}$', unit=r'', axis_type='log', force_log=True)
params.append(preLangevin)

R_series = FitParam(name='R_series', value=1e-4, bounds=[1e-5,1e-3], log_scale=True, value_type='float', fscale=None, rescale=False, display_name=r'$R_{series}$', unit=r'$\Omega$ m$^2$', axis_type='log', force_log=True)
params.append(R_series)

pnames = [param.full_name for param in params if param.type != "fixed"]

# Load the UltraNest run corresponding to the configured parameter list.
directory = os.path.join(os.path.dirname(__file__), "logs/loggauss/run1")
run, info = ultranest.read_file(directory, x_dim=len(pnames))

# Fill in missing parameter metadata when the saved run does not include it.
if "paramnames" not in info and "weighted_samples" in info:
    ndim = info["weighted_samples"]["points"].shape[1]
    info["paramnames"] = [f"x{i}" for i in range(ndim)]
if "paramnames_latex" not in info and "paramnames" in info:
    info["paramnames_latex"] = info["paramnames"]

# Convert samples to an ArviZ-compatible posterior dataset.
results_df = pd.DataFrame(data=info["samples"], columns=pnames)
results_df["chain"] = 0
results_df["draw"] = np.arange(len(results_df), dtype=int)
results_df = results_df.set_index(["chain", "draw"])

xdata = xr.Dataset.from_dataframe(results_df)
trace = az.InferenceData(posterior=xdata)

# Plot trace diagnostics and pairwise posterior structure.
az.plot_trace(trace)
az.plot_pair(trace, kind="kde", marginals=True)
plt.show()