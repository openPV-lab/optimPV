# This script is designed to run Bayesian inference using the UltraNest library for parameter estimation  in parallel using the MPI 
# see https://johannesbuchner.github.io/UltraNest/performance.html
# use the following command to run the script in parallel:
# mpiexec -np 10 python3 Ultranest_JV.py --num_live_points=400 

# Import necessary libraries
import warnings, os, sys, shutil
# remove warnings from the output
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import copy, uuid
import pySIMsalabim as sim
from pySIMsalabim.experiments.JV_steady_state import *
from ultranest import ReactiveNestedSampler
import argparse

try:
    from optimpv import *
except Exception as e:
    sys.path.append('../..') # add the path to the optimpv module
    from optimpv import *



# define command line arguments:
parser = argparse.ArgumentParser()


parser.add_argument("--num_live_points", type=int, default=400)
parser.add_argument('--log_dir', type=str, default='logs/log_JV')
parser.add_argument('--max_ncalls', type=int, default=int(1e5))

args = parser.parse_args()

num_live_points = args.num_live_points
log_dir = args.log_dir
max_ncalls = args.max_ncalls


# -----------------------------------------------------------------------------
# Define parameters to be optimized
# -----------------------------------------------------------------------------
params = [] # list of parameters to be optimized

mun = FitParam(name = 'l2.mu_n', value = 7e-8, bounds = [1e-9,1e-6], log_scale = True, value_type = 'float', fscale = None, rescale = False, display_name=r'$\mu_n$', unit='m$^2$ V$^{-1}$s$^{-1}$', axis_type = 'log', force_log = True)
params.append(mun)

mup = FitParam(name = 'l2.mu_p', value = 1e-8, bounds = [1e-9,1e-6], log_scale = True, value_type = 'float', fscale = None, rescale = False, display_name=r'$\mu_p$', unit=r'm$^2$ V$^{-1}$s$^{-1}$', axis_type = 'log', force_log = True)
params.append(mup)

# bulk_tr = FitParam(name = 'l2.N_t_bulk', value = 1e20, bounds = [1e19,1e22], log_scale = True, value_type = 'float', fscale = None, rescale = False,  display_name=r'$N_{T}$', unit=r'm$^{-3}$', axis_type = 'log', force_log = True)
# params.append(bulk_tr)

preLangevin = FitParam(name = 'l2.preLangevin', value = 1e-1, bounds = [0.005,1], log_scale = True, value_type = 'float', fscale = None, rescale = False, display_name=r'$\gamma_{pre}$', unit=r'', axis_type = 'log', force_log = True)
params.append(preLangevin)

R_series = FitParam(name = 'R_series', value = 1e-4, bounds = [1e-5,1e-3], log_scale = True, value_type = 'float', fscale = None, rescale = False,  display_name=r'$R_{series}$', unit=r'$\Omega$ m$^2$', axis_type = 'log', force_log = True)
params.append(R_series)

N_c = FitParam(name = 'l2.N_c', value = 1e27, bounds = [5e26,5e27], log_scale = True, value_type = 'float', fscale = None, rescale = False,  display_name=r'$N_{c}$', unit=r'm$^{-3}$', axis_type = 'log', force_log = True)
params.append(N_c)

# save the original parameters for later
params_orig = copy.deepcopy(params)

# -----------------------------------------------------------------------------
# Define target data
# -----------------------------------------------------------------------------
# Set the session path for the simulation and the input files
session_path = os.path.join(os.path.join(os.path.abspath('../'),'SIMsalabim','SimSS'))
input_path = os.path.join(os.path.join(os.path.join(os.path.abspath('../'),'Data','simsalabim_test_inputs','fakeOPV')))
simulation_setup_filename = 'simulation_setup_fakeOPV.txt'
simulation_setup = os.path.join(session_path, simulation_setup_filename) 

# path to the layer files defined in the simulation_setup file
l1 = 'ZnO.txt'
l2 = 'ActiveLayer.txt'
l3 = 'BM_HTL.txt'
l1 = os.path.join(input_path, l1)
l2 = os.path.join(input_path, l2)
l3 = os.path.join(input_path, l3)

# copy this files to session_path
force_copy = True
if not os.path.exists(session_path):
    os.makedirs(session_path)
for file in [l1,l2,l3,simulation_setup_filename]:
    file = os.path.join(input_path, os.path.basename(file))
    if force_copy or not os.path.exists(os.path.join(session_path, os.path.basename(file))):
        shutil.copyfile(file, os.path.join(session_path, os.path.basename(file)))
    else:
        print('File already exists: ',file)



# reset simss
# Set the JV parameters
Gfracs = [0.1,0.5,1] # Fractions of the generation rate to simulate (None if you want only one light intensity as define in the simulation_setup file)
UUID = str(uuid.uuid4()) # random UUID to avoid overwriting files

cmd_pars = [] # see pySIMsalabim documentation for the command line parameters
# Add the parameters to the command line arguments
for param in params:
    cmd_pars.append({'par':param.name, 'val':str(param.value)})

# Run the JV simulation
ret, mess = run_SS_JV(simulation_setup, session_path, JV_file_name = 'JV.dat', G_fracs = Gfracs, parallel = True, max_jobs = 3, UUID=UUID, cmd_pars=cmd_pars)

# save data for fitting
X,y = [],[]
X_orig,y_orig = [],[]
if Gfracs is None:
    data = pd.read_csv(os.path.join(session_path, 'JV_'+UUID+'.dat'), sep=r'\s+') # Load the data
    Vext = np.asarray(data['Vext'].values)
    Jext = np.asarray(data['Jext'].values)
    G = np.ones_like(Vext)
    rng = default_rng()#
    noise = rng.standard_normal(Jext.shape) * 0.01 * Jext
    Jext = Jext + noise
    X = Vext
    y = Jext

    # plt.figure()
    # plt.plot(X,y)
    # plt.show()
else:
    for Gfrac in Gfracs:
        data = pd.read_csv(os.path.join(session_path, 'JV_Gfrac_'+str(Gfrac)+'_'+UUID+'.dat'), sep=r'\s+') # Load the data
        Vext = np.asarray(data['Vext'].values)
        Jext = np.asarray(data['Jext'].values)
        G = np.ones_like(Vext)*Gfrac
        rng = default_rng()#
        noise = rng.standard_normal(Jext.shape) * 0.005 * Jext

        if len(X) == 0:
            X = np.vstack((Vext,G)).T
            y = Jext + noise
            y_orig = Jext 
        else:
            X = np.vstack((X,np.vstack((Vext,G)).T))
            y = np.hstack((y,Jext+ noise))
            y_orig = np.hstack((y_orig,Jext))

    # remove all the current where Jext is higher than a given value
    X = X[y<200]
    X_orig = copy.deepcopy(X)
    y_orig = y_orig[y<200]
    y = y[y<200]
    
    

    # plt.figure()
    # for Gfrac in Gfracs:
    #     plt.plot(X[X[:,1]==Gfrac,0],y[X[:,1]==Gfrac],label='Gfrac = '+str(Gfrac))
    # plt.xlabel('Voltage [V]')
    # plt.ylabel('Current density [A/m$^2$]')
    # plt.legend()
    # plt.show()

# -----------------------------------------------------------------------------
# Define Agent and log-likelihood for UltraNest
# -----------------------------------------------------------------------------

# Define the Agent and the target metric/loss function
from optimpv.models.DDfits.JVAgent import JVAgent
metric = 'mse' # can be 'nrmse', 'mse', 'mae'
loss = 'linear' # can be 'linear', 'huber', 'soft_l1'

# create a different params list for the agent
jv = JVAgent(params, X, y, session_path, simulation_setup, parallel = True, max_jobs = 3, metric = metric, loss = loss)

agents = [jv]
# Calulate the target metric for the original parameters
# best_fit_possible = loss_function(calc_metric(y,y_orig, metric_name = metric),loss)
# print('Best fit: ',best_fit_possible)

# -----------------------------------------------------------------------------
# Build search space from params
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Build search space from params
# -----------------------------------------------------------------------------

def create_search_space(params):
    x0 = []
    bounds = []
    param_mapping = []
    log_params_indices = []

    for param in params:
        if param.type == "fixed":
            continue

        param_mapping.append(param.name)
        current_index = len(x0)

        if param.value_type != "float":
            raise ValueError(
                f"Unsupported parameter type: {param.value_type}. Only 'float' is supported."
            )

        if param.force_log:
            if param.bounds[0] <= 0 or param.bounds[1] <= 0 or param.value <= 0:
                raise ValueError(f"Parameter {param.name} must be > 0 for log sampling.")
            log_params_indices.append(current_index)
            x0.append(np.log10(param.value))
            bounds.append((np.log10(param.bounds[0]), np.log10(param.bounds[1])))
        else:
            scale_factor = getattr(param, "fscale", 1.0) or 1.0
            x0.append(param.value / scale_factor)
            bounds.append((param.bounds[0] / scale_factor, param.bounds[1] / scale_factor))

    return np.asarray(x0), bounds, param_mapping, log_params_indices


x0, bounds, param_mapping, log_params_indices = create_search_space(params)
ndim = len(x0)

if ndim == 0:
    raise ValueError("No optimizable parameters were found.")
# -----------------------------------------------------------------------------
# Prior transform
# -----------------------------------------------------------------------------

def prior_transform(cube):
    theta = np.empty(ndim)

    for i, (lower, upper) in enumerate(bounds):
        theta[i] = lower + cube[i] * (upper - lower)

    return theta


# -----------------------------------------------------------------------------
# Log likelihood
# -----------------------------------------------------------------------------

all_metrics = []
for agent in agents:
    all_metrics.extend(agent.all_agent_metrics)


def loglike(theta):
    theta = np.asarray(theta, dtype=float)

    param_dict = {}
    idx = 0
    for param in params:
        if param.type == "fixed":
            param_dict[param.name] = param.value
        else:
            # Keep values in optimization space.
            # JVAgent / SIMsalabimAgent will apply 10** or fscale internally.
            param_dict[param.name] = theta[idx]
            idx += 1

    total_log_like = 0.0

    try:
        all_results = {}

        for agent in agents:
            all_results.update(agent.run_Ax(param_dict))

        for metric_name in all_metrics:
            if metric_name not in all_results:
                return -np.inf

            loss_val = all_results[metric_name]
            if not np.isfinite(loss_val):
                return -np.inf

            total_log_like += -0.5 * loss_val

        return total_log_like

    except Exception:
        return -np.inf


# -----------------------------------------------------------------------------
# Run UltraNest
# -----------------------------------------------------------------------------

sampler = ReactiveNestedSampler(
    param_mapping,
    loglike,
    transform=prior_transform,
    log_dir=log_dir,
)

result = sampler.run(
    show_status=True,
    max_ncalls=max_ncalls,
    min_num_live_points=num_live_points,
)

# -----------------------------------------------------------------------------
# Posterior samples
# -----------------------------------------------------------------------------

samples_opt = np.asarray(result["samples"])

samples_orig = np.empty_like(samples_opt)

for i, name in enumerate(param_mapping):

    param = next(p for p in params if p.name == name)

    if i in log_params_indices:
        samples_orig[:, i] = 10**samples_opt[:, i]
    else:
        scale_factor = getattr(param, "fscale", 1.0) or 1.0
        samples_orig[:, i] = samples_opt[:, i] * scale_factor

# -----------------------------------------------------------------------------
# Parameter summaries
# -----------------------------------------------------------------------------

results = {}

for i, name in enumerate(param_mapping):

    q16, q50, q84 = np.percentile(samples_orig[:, i], [16, 50, 84])

    results[name] = {
        "median": q50,
        "16th": q16,
        "84th": q84,
        "lower_err": q50 - q16,
        "upper_err": q84 - q50,
    }

    print(
        f"{name}: {q50:.4g} "
        f"(+{q84-q50:.3g} / -{q50-q16:.3g})"
    )

# -----------------------------------------------------------------------------
# Maximum-likelihood parameters
# -----------------------------------------------------------------------------

best_point_opt = np.asarray(
    result["maximum_likelihood"]["point"]
)

best_params = {}

for i, param in enumerate(
    [p for p in params if p.type != "fixed"]
):

    value = best_point_opt[i]

    if i in log_params_indices:
        value = 10**value
    else:
        scale_factor = getattr(param, "fscale", 1.0) or 1.0
        value = value * scale_factor

    best_params[param.name] = value

print("\nBest-fit parameters:")
print(best_params)


sampler.print_results()
sampler.plot()