Change Log
==========
All notable changes to this project will be documented in this file.

v1.05 - 2026-03-11 - VMLC-PV
-----------------------------
- axBOtorchOptimizer: skipping center and sobol as "models" when creating the ax client instance, this is to make sure that the client has a trainable model. Fixed small error in best Sobol value print. Also made sure that the SuggestOnly agent can work with SOBOL.
- SIMsalabimAgent: Remove the check that enforces left-to-right or right-to-left arrangement of the energy level offsets (differences in designated energy levels). This is not necessary, as a later check ensures that each parameter is specified only once. Fix some small bugs.   
- JVAgent: Added new exp_format ‘^QFLSL-?\d+$’ i.e. QFLSL plus an integer. This is to simulate QFLS vs Vext curves. Added a G_eff parameter that get multiplied to the G_frac this helps when using the transfer matrix model ('calc' option in SIMsalabim) and the n,k data are not perfect. Small fix for the Voc calculation in the Kirchartz QFLS version.
- RateEqModel: Removed a factor 2 in the p1s expression. Added N0 as a potential fit parameter.  
- scipyOptimizer: Added optimize_least_squares method to perform optimization using the least squares method from scipy. This is useful for fitting problems and can provide better results than the standard scipy.optimize.minimize method for certain problems.
- posterior: Added new possibility to calculate the approximate posterior distributions with approx_posteriot.py and lazy_posterior.py. posterior.py will be retired soon. exploration_density.py was created to keep the density exploration plotting functions.
- general.py: Changed the way to specify transformations for the data. We used to have compare_type in all the agents. Now we can specify transformations as a string or a list of strings in the transform_data function. The available transforms are: 'log', 'linear', 'abs', and 'normalize'. The default is 'linear' if nothing is specified. This change was made to make the code more consistent and easier to use across all agents.
- New data and Notebooks for testing the JVAgent with QFLS vs Vext curves.
- New JVSweepAgent to simulate JV sweeps with the SIMsalabimAgent. 
- New and updated Notebooks for the new functionalities.
- update requirements for ax-platform to 1.2.4
- Rearranged the optimizers,  models and Notebooks.
- Added three External Generation Nodes for ax. The TuRBOGenerationNode, MorboGenerationNode and REITuRBOGenerationNode. This allows to use these optimizers with the ax-platform in a more seamless way and to take advantage of the features of the ax-platform.
- Since step_size is implemented in the ax-platform I changed the way the stepsize argument was used. It is now possible to specify stepsize also for float parameters and not just for intergers. The stepsize argument is now passed to the ax client instance and is used by the ax-platform to determine the step size for the optimization process. This allows for more flexibility in the optimization process and can lead to better results.
- Added 'update_version_param_files.py' script to update the version number in the parameter files used by SIMsalabim and stored in the Data folder.

v1.04 - 2025-09-25 - VMLC-PV
-----------------------------
- DiodeAgent: Updated the code to use the transform_data function to make it consistent with the other Agents.
- JVAgent, RateEqAgent general: modified the transform_data function to make the transformation based on the G_frac when needed for the JVAgent and the RateEqAgent. In those cases, the G_frac column is assumed to be the second column of the X array. 
- axBOtorchOptimizer: Added support for inequality constraints in optimize_turbo. I also added an acceleration parameter to the TurboState to control the expansion or contraction of the trust region. In the original paper, it is set to 2, and I now set it to 1.6 (i.e. roughly the Golden ratio).
- RateEqAgent: Added the possibility to use the ratio between C_n and C_p instead of using C_p as a fitting parameter. The new parameter is called ratio_Cnp and is defined such that ratio_Cnp = C_n/C_p.  
Added support for parallelization using joblib to run the different G_frac in parallel. This can significantly speed up the fitting process when multiple G_frac are used. The number of parallel jobs can be controlled using the max_jobs parameter.
- Data: Added some new experimental data for testing the RateEqAgent and the DBTD_multi_trap model based on the work by Maxim Simmonds (HZB) and the repository [MAPI-FAPI-fitting](https://github.com/MaximSimmonds-HZB/MAPI-FAPI-fitting).
- Updated tests for the new functionalities.
- Updated the notebooks to use the new functionalities.
- JVAgent: Removed bug in ambi_param_transform for N_ions with force_log=True.

v1.03 - 2025-08-14 - VMLC-PV
-----------------------------
- Upgrading the ax-platform version to 1.0.0.
- axBOtorchOptimizer: Removing the optimize(batch=True) that used the runner in older versions of optimPV. Mostly because it did not provide any significant benefits and added unnecessary complexity and that the new ax-platform broke the compatibility with the old code. Multiple small fixes to the code to make it compatible with the new ax-platform version 1.0.0. Removed the parallel_agent option and replaced it with a parallel option that will run everything in parallel.
- PymooOptimizer: Initial implementation of the PymooOptimizer class for single and multi-objective optimization using the [pymoo](https://github.com/anyoptimization/pymoo) library. This class provide new optimization capabilities withe the evolutionary/genetic algorithms and other algorithms available in the Pymoo library. The class is designed to have a similar interface to the AxOptimizer class, allowing for easy integration with existing code. The class supports both single and multi-objective optimization, and can be used with the same parameters and agents as the AxOptimizer class. The class also supports parallel optimization using the Pymoo library's parallelization capabilities. However, for not it only support float as value_type for the parameters. 
- posterior: added the 'pymoo' optimizer as an options for the **plot_density_exploration** function. This allows to plot the density exploration of the optimization process using the Pymoo library's optimization results.
- New Notebooks and created tests for the new functionalities.
- logger: Added logger inspired but the one in the Ax library. 
- SuggestOnlyAgent: Added the SuggestOnlyAgent class to provide a simple way to suggest new parameters without running the simulation/experiment and using a known dataset. This is useful when doing design of experiments (DoE) in multiple steps. It can be used with the axBOtorchOptimizer and the PymooOptimizer classes to suggest new parameters based on the previous results. 
- EGBO: Updated the EGBOAcquisition class to work with the new ax-platform version 1.0.0. 
- RateEqAgent: small fixes for the case Gfracs = None we where not taking the right time axis to take into account the pump frequency, this was done properly when we had several Gfracs. 
- RateEqModel: Added two new model based on the work by [Kober-Czerny et al. 2025](https://doi.org/10.1103/PRXEnergy.4.013001) and [M. Simmonds](https://github.com/MaximSimmonds-HZB/MAPI-FAPI-fitting).
- Pumps: fixing the initial_carrier_density to convert the N0 into proper generation rate.

v1.02 - 2025-05-20 - VMLC-PV
-----------------------------
- Fixing the ax-platform version to 0.5.0 as version 1.0.0 breaks the compatibility with the current code. This is a temporary fix until the code is updated to be compatible with ax-platform 1.0.0.
- scipyOptimizer: Adding the scipyOptimizer class to provide a simple interface for optimization using the scipy library. The class is typically for single-objective optimization problems and supports various optimization algorithms available in the [scipy](https://github.com/scipy/scipy) library.
- EmceeOptimizer: Adding the EmceeOptimizer class to provide an interface for optimization using the [emcee](https://github.com/dfm/emcee) library. The class is designed to work with the Bayesian inference methods provided by emcee, allowing for efficient sampling of the parameter space.

v1.01 - 2025-04-29 - VMLC-PV
-----------------------------
- BaseAgent.py: Modified the params_w so such that param.type = 'fixed' are not transformed.
- JVAgent.py, IMPSAgent,py, ImpedanceAgent.py, HysteresisAgent.py, CVAgent.py: Added a safety check to ensure there is no infinite loop in the __init__ method when checking for the input simulation files for SIMsalabim.
- Small docstrings fixes.


v1 - 2025-04-10 - VMLC-PV
---------------------------
- Initial release of the project.