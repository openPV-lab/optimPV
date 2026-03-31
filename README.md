

<img src="./docs/logo/optimPV_logo.png" alt="optimPV logo" width="100px">  

# optimPV: Optimization & Modeling tools for PV research

## Authors
[Vincent M. Le Corre](https://github.com/VMLC-PV)  
[Filip Fekete](https://github.com/FilipFekete)
[Larry Lüer](https://github.com/larryluer)

## Institution
<img src="./docs/logo/sdu_logo.jpg" alt="SDU logo" width="100px"> CAPE - Centre for Advanced Photovoltaics and Thin-film Energy Devices, University of Southern Denmark, Denmark

## Description
This repository contains the code to run **optimPV**. optimPV combines sevral optimization procedures and modeling utilities that can be used for two objectives:
1. to optimize the parameters of a simulation to fit experimental data.
2. to optimize the processing conditions in a self-driving experimental set-up.  

## Repository Folder Structure
    .
    ├── Main                             # Main directory
        ├── Data                         # Contains some example data for the notebooks
        ├── Notebooks                    # Contains clean versions of the Notebooks
        ├── docs                         # Contains the documentation
        ├── optimPV                      # Main directory for the optimPV codes
            ├── general                  # Directory with general utility functions used by the different agents
            ├── optimizers                # Directory with the different optimizers
                ├── BayesInfEmcee            # Directory with the Bayesian inference codes using emcee
                ├── axBOtorch                # Directory with the Bayesian optimization (BO) codes using BoTorch and Ax
                ├── pymooOpti               # Directory with the optimization codes using pymoo
                ├── scipyOpti                # Directory with the optimization codes using scipy.optimize
            ├── models                    # Directory with the different models used by the agents
                ├── DDfits                   # Directory with the different agents to run the drift-diffusion simulator  [SIMsalabim](https://github.com/kostergroup/SIMsalabim) for JV, Hysteresis, Impedance, CV and IMPS simulations and fitting
                ├── Diodefits                # Directory with the agent to simulate and fit the non-ideal diode equation model
                ├── RateEqfits               # Directory with the agent to simulate and fit the rate equations for different experiment types (trPL, TRMC, TAS, etc.)
                ├── TransferMatrix           # Directory with the agent to run the transfer matrix simulations
            ├── posterior                # Directory with some utility functions to plot the posterior distributions using the BO surrogate model
            ├── tests                         # Contains the codes for testing optimPV
    └── README.md

## Installation
### With pip
To install optimPV with pip you have two options:
1. Install optimPV using the [PyPI repository](https://pypi.org/project/optimpv/)  

    ```bash
    pip install optimpv
    ```

2. Install optimPV using the GitHub repository   https://github.com/openPV-lab/optimPV

    ```bash
    pip install git+https://github.com/openPV-lab/optimPV
    ```

### With conda
To install optimPV with conda:

```bash
conda create -n optimpv 
conda activate optimpv
pip install optimpv
```

You can also clone your base environment:

```bash  
conda create -n optimpv --clone base
```

## Additional necessary installs for the agents
### Drift-diffusion agent
The drift-diffusion agent uses [SIMsalabim](https://github.com/kostergroup/SIMsalabim) to run drift-diffusion simulations.

- SIMsalabim is included as a submodule.
- Install it following the instructions on the [SIMsalabim GitHub repository](https://github.com/kostergroup/SIMsalabim).
- Only works for parallel simulations on Linux. All other optimPV agents work on Windows.

### Parallel simulations
To run parallel simulations on Linux you can also install GNU Parallel:

```bash
sudo apt update
sudo apt install parallel
```

## Disclaimer
This repository is still under development. If you find any bugs or have any questions, please contact us.
