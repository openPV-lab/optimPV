"""Rate equation models for charge carrier dynamics in semiconductors"""
# Note: This class is inspired by the https://github.com/i-MEET/boar/ package
######### Package Imports #########################################################################

import warnings,time,re
import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.sparse import lil_matrix
from functools import partial
from logging import Logger

from optimpv.general.logger import get_logger, _round_floats_for_logging

logger: Logger = get_logger('RateEqModel')
ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)

## Physics constants
from scipy import interpolate, constants
kb = constants.value(u'Boltzmann constant in eV/K')

######### Function Definitions ####################################################################
def BT_model(parameters, t, Gpulse, t_span, N0=0, G_frac = 1,  equilibrate=True, eq_limit=1e-2, maxcount=1e3, solver_func = 'solve_ivp', **kwargs):
    """Solve the bimolecular trapping equation :  
    
    dn/dt = G - k_trap * n - k_direct * n^2
    
    Based on the beautiful work of:

    Péan, Emmanuel V. and Dimitrov, Stoichko and De Castro, Catherine S. and Davies, Matthew L., 
    Phys. Chem. Chem. Phys., 2020,22, 28345-28358, http://dx.doi.org/10.1039/D0CP04950F

    Parameters
    ----------
    parameters : dict
        dictionary containing the parameters of the model it must contain 'k_trap' and 'k_direct'.
            'k_trap' : float
                trapping rate constant in s^-1
            'k_direct' : float
                Bimolecular/direct recombination rate constant in m^-3 s^-1

    t : ndarray of shape (n,)
        array of time values

    G :  ndarray of shape (n,)
        array of values of the charge carrier generation rate m^-3 s^-1

    t_span : ndarray of shape (n,), optional
        array of time values for the pulse time step in case it is different from t, by default None

    N0 : float, optional
        initial value of the charge carrier density, by default 0

    G_frac : float, optional
        fraction of the generation rate that is absorbed, by default 1
    
    equilibrate : bool, optional
        make sure equilibrium is reached?, by default True
    
    eq_limit : float, optional
        relative change of the last time point to the previous one, by default 1e-2
    
    maxcount : int, optional
        maximum number of iterations to reach equilibrium, by default 1e3
    
    solver_func : str, optional
        solver function to use can be ['odeint','solve_ivp'], by default 'solve_ivp'

    kwargs : dict
        additional keyword arguments for the solver function
            'method' : str, optional
                method to use for the solver, by default 'RK45'
            'rtol' : float, optional
                relative tolerance, by default 1e-3
    
    Returns
    -------
    ndarray of shape (n,)
        array of values of the charge carrier density m^-3

    """   
    if 'k_trap' in parameters.keys():
        k_trap = parameters['k_trap']
    else:
        raise ValueError('k_trap is not in the parameters dictionary')
    
    if 'k_direct' in parameters.keys():
        k_direct = parameters['k_direct']
    else:
        raise ValueError('k_direct is not in the parameters dictionary')
    
    # check solver function
    if solver_func not in ['odeint','solve_ivp']:
        warnings.warn('solver function not recognized, using odeint', UserWarning)
        solver_func = 'odeint'

    # kwargs
    method = kwargs.get('method', 'RK45')
    rtol = kwargs.get('rtol', 1e-6)

    # check if the pulse time step is different from the time vector
    if t_span is None:
        t_span = t

    def dndt(t, y, t_span, Gpulse, k_trap, k_direct):
        """Bimolecular trapping equation
        """  
        gen = np.interp(t, t_span, Gpulse) # interpolate the generation rate at the current time point
        
        S = gen - k_trap * y - k_direct * y**2
        return S.T

    # Solve the ODE
    if equilibrate: # make sure the system is in equilibrium 
        # to be sure we equilibrate the system properly we need to solve the dynamic equation over the full range of 1/fpu in time
        rend = 1e-20 # last time point
        RealChange = 1e19 # initialize the relative change with a high number
        rstart = N0*G_frac+rend
        count = 0
        while np.any(abs(RealChange) > eq_limit) and count < maxcount:
            if solver_func == 'odeint':
                r = odeint(dndt, rstart, t_span, args=(t_span, Gpulse, k_trap, k_direct), tfirst=True, **kwargs)
                RealChange = (r[-1] -rend)/rend # relative change of mean
                rend = r[-1] # last time point
            elif solver_func == 'solve_ivp':
                # r = solve_ivp(dndt, [t[0], t[-1]], rstart, args=(t_span, Gpulse, k_trap, k_direct), method = method, rtol=rtol)
                r = solve_ivp(partial(dndt,t_span = t_span, Gpulse = Gpulse, k_trap = k_trap, k_direct = k_direct), [t[0], t[-1]], [N0*G_frac], t_eval = t, method = method, rtol=rtol)
    
                RealChange  = (r.y[:,-1] -rend)/rend # relative change of mean
                rend = r.y[:,-1] # last time point
            rstart = N0+rend
            count += 1

    else:
        rstart = N0
    
    # solve the ODE again with the new initial conditions with the equilibrated system and the original time vector
    Gpulse_eq = np.interp(t, t_span, Gpulse) # interpolate the generation rate at the current time point
    if solver_func == 'odeint':
        r = odeint(dndt, rstart, t, args=(t, Gpulse_eq, k_trap, k_direct), tfirst=True, **kwargs)
        return r[:,0], r[:,0]
    elif solver_func == 'solve_ivp':
        # r = solve_ivp(dndt, [t[0], t[-1]], rstart, t_eval = t, args=(t, Gpulse_eq, k_trap, k_direct), method = method, rtol=rtol)
        r = solve_ivp(partial(dndt,t_span = t, Gpulse = Gpulse_eq, k_trap = k_trap, k_direct = k_direct), [t[0], t[-1]], rend + N0*G_frac, t_eval = t, method = method, rtol=rtol)

        # return n and p concentrations (they are the same)
        return r.y[0] , r.y[0]


def BTD_model(parameters, t, Gpulse, t_span, N0=0, G_frac = 1, equilibrate=True, eq_limit=1e-2,maxcount=1e3, solver_func = 'odeint', output_trap_dens = False,**kwargs):
    """Solve the bimolecular trapping and detrapping equation :

    dn/dt = G - k_trap * n * (N_t_bulk - n_t) - k_direct * n * (p + N_A)
    dn_t/dt = k_trap * n * (N_t_bulk - n_t) - k_detrap * n_t * (p + N_A)
    dp/dt = G - k_detrap * n_t * (p + N_A) - k_direct * n * (p + N_A)

    Based on the beautiful work of:

    Péan, Emmanuel V. and Dimitrov, Stoichko and De Castro, Catherine S. and Davies, Matthew L., 
    Phys. Chem. Chem. Phys., 2020,22, 28345-28358, http://dx.doi.org/10.1039/D0CP04950F
    
    Parameters
    ----------
    parameters : dict
        dictionary containing the parameters of the model it must contain 'k_trap', 'k_direct', 'k_detrap', 'N_t_bulk' and 'N_A'.

            k_trap : float
                Trapping rate constant in m^3 s^-1
            k_direct : float
                Bimolecular/direct recombination rate constant in m^3 s^-1
            k_detrap : float
                Detrapping rate constant in m^3 s^-1
            N_t_bulk : float
                Bulk trap density in m^-3
            N_A : float
                Ionized p-doping concentration in m^-3

    t : ndarray of shape (n,)
        time values in s
    Gpulse : ndarray of shape (n,)
        array of values of the charge carrier generation rate m^-3 s^-1
    t_span : ndarray of shape (n,), optional
        time values for the pulse time step in case it is different from t, by default None
    N0 : float, optional
        initial values of the electron, trapped electron and hole concentrations in m^-3, by default 0
    G_frac : float, optional
        fraction of the generation rate that is absorbed, by default 1
    equilibrate : bool, optional
        whether to equilibrate the system, by default True
    eq_limit : float, optional
        limit for the relative change of the last time point to the previous one to consider the system in equilibrium, by default 1e-2
    maxcount : int, optional
        maximum number of iterations to reach equilibrium, by default 1e3
    solver_func : str, optional
        solver function to use can be ['odeint','solve_ivp'], by default 'odeint'
    output_trap_dens : bool, optional
        whether to output the trapped electron concentration, by default False
    kwargs : dict
        additional keyword arguments for the solver function

            'method' : str, optional
                method to use for the solver, by default 'RK45'
            'rtol' : float, optional
                relative tolerance, by default 1e-3

    Returns
    -------
    ndarray of shape (n,)
        electron concentration in m^-3
    ndarray of shape (n,)
        hole concentration in m^-3
    ndarray of shape (n,)
        if output_trap_dens is True then we also output trapped electron concentration in m^-3

    """   
    if 'k_trap' in parameters.keys():
        k_trap = parameters['k_trap']
    else:
        raise ValueError('k_trap is not in the parameters dictionary')
    
    if 'k_direct' in parameters.keys():
        k_direct = parameters['k_direct']   
    else:
        raise ValueError('k_direct is not in the parameters dictionary')
    
    if 'k_detrap' in parameters.keys():
        k_detrap = parameters['k_detrap']
    else:
        raise ValueError('k_detrap is not in the parameters dictionary')
    
    if 'N_t_bulk' in parameters.keys():
        N_t_bulk = parameters['N_t_bulk']
    else:
        raise ValueError('N_t_bulk is not in the parameters dictionary')
    
    if 'N_A' in parameters.keys():
        N_A = parameters['N_A']
    else:
        N_A = 0
        # warnings.warn('N_A is not in the parameters dictionary so it will be set to 0', UserWarning)
        # raise ValueError('N_A is not in the parameters dictionary')
    
    # check solver function
    if solver_func not in ['odeint','solve_ivp']:
        warnings.warn('solver function not recognized, using odeint', UserWarning)
        solver_func = 'odeint'

    # kwargs
    method = kwargs.get('method', 'RK45')
    rtol = kwargs.get('rtol', 1e-3)

    # check if the pulse time step is different from the time vector
    if t_span is None:
            t_span = t
    N_init = [N0, 0, N0] # initial conditions
    def rate_equations(t, n, t_span, Gpulse, k_trap, k_direct, k_detrap, N_t_bulk, N_A):
            """Rate equation of the BTD model (PEARS) 

            Parameters
            ----------
            t : float
                time in s
            n : list of floats
                electron, trapped electron and hole concentrations in m^-3
            Gpulse : ndarray of shape (n,)
                array of values of the charge carrier generation rate m^-3 s^-1
            t_span : ndarray of shape (n,), optional
                array of time values for the pulse time step in case it is different from t, by default None
            k_trap : float
                trapping rate constant in m^3 s^-1
            k_direct : float
                Bimolecular/direct recombination rate constant in m^3 s^-1
            k_detrap : float
                detrapping rate constant in m^3 s^-1
            N_t_bulk : float
                bulk trap density in m^-3
            N_A : float
                ionized p-doping concentration in m^-3

            Returns
            -------
            list
                Fractional change of electron, trapped electron and hole concentrations at times t
            """

            gen = np.interp(t, t_span, Gpulse) # interpolate the generation rate at the current time point
            
            n_e, n_t, n_h = n
            
            B = k_direct * n_e * (n_h + N_A)
            T = k_trap * n_e * (N_t_bulk - n_t)
            D = k_detrap * n_t * (n_h + N_A)
            dne_dt = gen - B - T
            dnt_dt = T - D
            dnh_dt = gen - B - D
            return [dne_dt, dnt_dt, dnh_dt]

    # Solve the ODE
    if equilibrate: # equilibrate the system
        # to be sure we equilibrate the system properly we need to solve the dynamic equation over the full range of 1/fpu in time 
        rend = [1e-20,1e-20,1e-20] # initial conditions
        rstart = [rend[0] + N0*G_frac, rend[1] , rend[2] + N0*G_frac] # initial conditions for the next integration
        RealChange = 1e19 # initialize the relative change with a high number
        count = 0
        while np.any(abs(RealChange) > eq_limit) and count < maxcount:

            if solver_func == 'solve_ivp':
                r = solve_ivp(partial(rate_equations,t_span = t_span, Gpulse = Gpulse, k_trap = k_trap, k_direct = k_direct, k_detrap = k_detrap, N_t_bulk = N_t_bulk, N_A = N_A), [t[0], t[-1]], rstart, t_eval = None, method = method, rtol= rtol) # method='LSODA','RK45'
                # monitor only the electron concentration           
                RealChange  = (r.y[0,-1] - rend[0])/rend[0] # relative change of mean
                rend = [r.y[0,-1], r.y[1,-1], r.y[2,-1]] # last time point
            elif solver_func == 'odeint':
                r = odeint(rate_equations, rstart, t_span, args=(t_span, Gpulse, k_trap, k_direct, k_detrap, N_t_bulk, N_A), tfirst=True, rtol=rtol)
                RealChange = (r[-1,0]-rend[0])/rend[0] # relative change of mean
                rend = [r[-1,0], r[-1,1], r[-1,2]] # last time point

            rstart = [rend[0] + N0*G_frac, rend[1] , rend[2] + N0*G_frac] # initial conditions for the next integration
            count += 1
    else:
        rstart = [N0, 0, N0] 


    # solve the ODE again with the new initial conditions with the equilibrated system and the original time vector
    Gpulse_eq = np.interp(t, t_span, Gpulse) # interpolate the generation rate at the current time point
    if solver_func == 'solve_ivp':
        r = solve_ivp(partial(rate_equations,t_span = t, Gpulse = Gpulse_eq, k_trap = k_trap, k_direct = k_direct, k_detrap = k_detrap, N_t_bulk = N_t_bulk, N_A = N_A), [t[0], t[-1]], rstart, t_eval = t, method = method, rtol= rtol) # method='LSODA','RK45'
        n_e = r.y[0]
        n_t = r.y[1]
        n_h = r.y[2]
    elif solver_func == 'odeint':
        r = odeint(rate_equations, rstart, t, args=(t, Gpulse_eq, k_trap, k_direct, k_detrap, N_t_bulk, N_A), tfirst=True, rtol=rtol)
        n_e = r[:,0]
        n_t = r[:,1]
        n_h = r[:,2]

    if output_trap_dens:
        return n_e,  n_h, n_t
    else:
        # return electron and hole concentrations
        return n_e, n_h
    

def DBTD_model(parameters, t, Gpulse, t_span, N0=0, G_frac = 1, equilibrate=True, eq_limit=1e-2, maxcount=1e3, output_integrated_values = True,**kwargs):
    """Solve the diffusion bimolecular trapping and detrapping model including diffusion.  

    The rate equation and model used here are based on the work by [Kober-Czerny et al. 2025](https://doi.org/10.1103/PRXEnergy.4.013001) see the [paper](https://doi.org/10.1103/PRXEnergy.4.013001) for more details or the [GitHub repository](https://github.com/manuelkoberczerny/assessing-TRPL-with-bayesian-inference_and-MCMC).
    
    Parameters
    ----------
    parameters : dict
        dictionary containing the parameters of the model it must contain 'k_trap', 'k_direct', 'k_detrap', 'N_t_bulk' and 'N_A'.

            k_direct : float
                Bimolecular/direct recombination rate constant in m^3 s^-1
            k_deep : float
                Deep trap rate constant in s^-1
            k_c : float
                Capture rate constant in s^-1
            k_e : float
                Electron emission rate constant in s^-1
            S_front : float
                Front surface recombination velocity in m s^-1
            S_back : float
                Back surface recombination velocity in m s^-1
            N_A : float
                Acceptor doping density in m^-3
            L : float
                Length of the device in m
            alpha : float
                Absorption coefficient in m^-1
            mu : float
                Mobility in m^2 V^-1 s^-1
            T : float
                Temperature in K, by default 300

    t : ndarray of shape (n,)
        time values in s
    Gpulse : ndarray of shape (n,)
        array of values of the charge carrier generation rate m^-3 s^-1
    t_span : ndarray of shape (n,), optional
        time values for the pulse time step in case it is different from t, by default None
    N0 : float, optional
        initial values of the electron, trapped electron and hole concentrations in m^-3, by default 0
    G_frac : float, optional
        fraction of the generation rate that is absorbed, by default 1
    equilibrate : bool, optional
        whether to equilibrate the system, by default True
    eq_limit : float, optional
        limit for the relative change of the last time point to the previous one to consider the system in equilibrium, by default 1e-2
    maxcount : int, optional
        maximum number of iterations to reach equilibrium, by default 1e3
    output_integrated_values : bool, optional
        whether to output the integrated values, by default True
    kwargs : dict
        additional keyword arguments for the solver function
            'grid_size' : int, optional
                size of the grid for the spatial discretization, by default 100

    Returns
    -------
    list or ndarray
        The integrated values of the electron density versus time and space.
        Each element of the list corresponds to a specific time point and contains the electron density values at different spatial positions.
    list or ndarray
        The integrated values of the hole density versus time and space.
        Each element of the list corresponds to a specific time point and contains the hole density values at different spatial positions.

    Raises
    ------
    ValueError
        If the parameters are not valid.

    """       

    if 'S_front' in parameters.keys():
        S_front = parameters['S_front'] 
    else:
        raise ValueError('S_front is not in the parameters dictionary')
    if 'S_back' in parameters.keys():
        S_back = parameters['S_back']
    else:
        raise ValueError('S_back is not in the parameters dictionary')
    if 'mu' in parameters.keys():
        mu = parameters['mu']
    else:
        raise ValueError('mu is not in the parameters dictionary')
    if 'k_direct' in parameters.keys():
        k_direct = parameters['k_direct']
    else:
        raise ValueError('k_direct is not in the parameters dictionary')
    if 'k_deep' in parameters.keys():
        k_deep = parameters['k_deep']
    else:
        raise ValueError('k_deep is not in the parameters dictionary')
    if 'k_c' in parameters.keys():
        k_c = parameters['k_c']
    else:
        raise ValueError('k_c is not in the parameters dictionary')
    if 'k_e' in parameters.keys():
        k_e = parameters['k_e']
    else:
        raise ValueError('k_e is not in the parameters dictionary')
    if 'N_A' in parameters.keys():
        N_A = parameters['N_A']
    else:
        raise ValueError('N_A is not in the parameters dictionary')
    if 'alpha' in parameters.keys():
        alpha = parameters['alpha']
    else:
        raise ValueError('alpha is not in the parameters dictionary')
    if 'L' in parameters.keys():
        L = parameters['L']
    else:
        raise ValueError('L is not in the parameters dictionary')
    if 'T' in parameters.keys():
        T = parameters['T']
    else:
        T = 300 # default temperature in Kelvin

    N_init = N0 *G_frac# initial conditions

    grid_size = kwargs.get('grid_size', 100)  # number of grid points
    z_array = np.linspace(0, L, grid_size)  # spatial domain
    ds = z_array[1] - z_array[0]
    

    # gen = np.interp(t, t_span, Gpulse) # interpolate the generation rate at the current time point
    mean_beer_lambert = np.mean(np.exp(-alpha * z_array))  # Beer-Lambert law for generation profile
    generation = np.zeros((len(t_span), len(z_array)))
    for i in range(len(t_span)):
        generation[i] = Gpulse[i] * np.exp(-alpha * z_array) / mean_beer_lambert  # normalize the generation profile

    dt = np.diff(t_span)
    # Diffusion Coefficient in cm2 s-1
    # limit_mobility = (thickness * 1e-7)**2 / (abs(time[1] - time[0])) / (1.380649e-23 * 292 / 1.6021766e-19)  # cm2 (Vs)-1 Formula : L^2 / (t * kT/q)
    limit_mobility = (L**2 / abs(dt[1])) / (kb * T)  # m2 (Vs)-1 Formula : L^2 / (Delta t * kT/q)
    Diffusion_coefficient = mu * kb * T  # m2 s-1
    if mu >= limit_mobility / 4:
        alpha = 0


    def rate_equations(n_dens, nt, generation, k_direct, k_deep, k_c, k_e):           

        p_dens = n_dens + nt

        R_rad = - k_direct*n_dens*p_dens
        
        dnt_dt = k_c*n_dens - k_e*nt
        R_nr = - k_c*n_dens + k_e*nt - k_deep*n_dens
        
        dn_dt = R_rad + R_nr + generation
        # print('generation:', list(generation))
        # print('dn_dt:', list(R_rad + R_nr))
        
        return dn_dt, dnt_dt

    def Runge_Kutta_R4(n_dens, nt, generation, dt, k_direct, k_deep, k_c, k_e):

        RuKu1_n, RuKu1_nt = rate_equations(n_dens, nt, generation, k_direct, k_deep, k_c, k_e)
        RuKu2_n, RuKu2_nt = rate_equations(n_dens + RuKu1_n*dt/2, nt + RuKu1_nt*dt/2, generation, k_direct, k_deep, k_c, k_e)
        RuKu3_n, RuKu3_nt = rate_equations(n_dens + RuKu2_n*dt/2, nt + RuKu2_nt*dt/2, generation, k_direct, k_deep, k_c, k_e)
        RuKu4_n, RuKu4_nt = rate_equations(n_dens + RuKu3_n*dt, nt + RuKu3_nt*dt, generation, k_direct, k_deep, k_c, k_e)

        Ruku_n = (RuKu1_n + 2*RuKu2_n + 2*RuKu3_n + RuKu4_n)/6
        Ruku_nt = (RuKu1_nt + 2*RuKu2_nt + 2*RuKu3_nt + RuKu4_nt)/6

        return Ruku_n, Ruku_nt
    
    def X_n_maker(d_factor, size, dx, D, Sf, Sb):
        x_size = np.zeros((size.size, size.size))
        Xn_1 = np.diag(d_factor * np.ones(size.size - 1), -1)
        
        Xn_2 = np.diag((1 - 2 * d_factor) * np.ones(size.size))
        Xn_2[0, 0] = 1 - d_factor - (dx / D) * d_factor * Sf
        Xn_2[-1, -1] = 1 - d_factor - (dx / D) * d_factor * Sb
        
        Xn_3 = np.diag(d_factor * np.ones(size.size - 1), 1)
        
        return Xn_1 + Xn_2 + Xn_3
    
    def total_recombination_rate(dt_current, n_dens, p_dens, generation, ds, k_direct, k_deep, k_c, k_e, Diffusion_coefficient, S_front, S_back):
        # a. Recombination (Runge-Kutta Algorithm)
        nt = p_dens - n_dens
        Ruku_n, Ruku_nt  = Runge_Kutta_R4(n_dens, nt, generation, dt_current, k_direct, k_deep, k_c, k_e)
        # b. Diffusion
        d_factor = Diffusion_coefficient*dt_current/(2*ds*ds)
        A_n = X_n_maker(-d_factor, n_dens, ds, Diffusion_coefficient, S_front, S_back)
        B_n = X_n_maker(d_factor, n_dens, ds, Diffusion_coefficient, S_front, S_back)

        Bn_dot_n_dens = np.dot(B_n, n_dens) + Ruku_n*dt_current/2
        n_dens_new = np.linalg.solve(A_n, Bn_dot_n_dens)

        # c. Physical limits
        n_dens_new = np.where(n_dens_new <= 0, 0, n_dens_new)
        p_dens_new = n_dens_new + nt + Ruku_nt*dt_current
        p_dens_new = np.where(p_dens_new <= 0, 0, p_dens_new)
        
        return n_dens_new, p_dens_new
    
    # Initial Charge-Carrier Density
    n0_z = N_init * np.exp(-alpha * z_array) / np.mean(np.exp(-alpha * z_array))  # normalize the initial charge carrier density
    n_dens = np.zeros((len(t_span), len(z_array)))
    p_dens = np.zeros((len(t_span), len(z_array)))
    n_dens[0] = n0_z
    p_dens[0] = n0_z + N_A  # initial hole density

    # Looping over time-domain
    count = 0
    if equilibrate:  # equilibrate the system
        while True:
            if count != 0:
                n_dens[0] = n_dens[-1] + n0_z
                # print(f'n0: {n0_z}, n_dens[0]: {n_dens[0]}')
                p_dens[0] = p_dens[-1] + n0_z + N_A
            for i in range(1, len(t_span)):
                n_dens[i], p_dens[i] = total_recombination_rate(dt[i-1], n_dens[i-1], p_dens[i-1], generation[i], ds, k_direct, k_deep, k_c, k_e, Diffusion_coefficient, S_front, S_back) # not sure if the generation index should be [i] or [i-1]
            if count == 0:
                count += 1
                rend = n_dens[-1]
                # print(n_dens[-1])
            else:
                # print(n_dens[-1])
                diff = n_dens[-1] - rend
                if np.all(diff == 0):
                    break
                RealChange = (n_dens[-1] - rend) / rend
                rend = n_dens[-1]
                count += 1
                # print(f'Equilibrating: {count} iterations, Relative Change: {np.max(abs(RealChange))}')
                if np.all(abs(RealChange) < eq_limit) or count > maxcount or np.sum(diff == 0):
                    break

    # rerun for t this time
    n_dens = np.zeros((len(t), len(z_array)))
    p_dens = np.zeros((len(t), len(z_array)))
    n_dens[0] = n_dens[-1] + n0_z
    p_dens[0] = p_dens[-1] + n0_z + N_A
    dt = np.diff(t)
    for i in range(1, len(t)):
        n_dens[i], p_dens[i] = total_recombination_rate(dt[i-1], n_dens[i-1], p_dens[i-1], generation[i], ds, k_direct, k_deep, k_c, k_e, Diffusion_coefficient, S_front, S_back)



    if output_integrated_values:
        # get the densities in between grid points
        n_dens = (n_dens[:, 1:]+ n_dens[:, :-1])/2
        p_dens = (p_dens[:, 1:]+ p_dens[:, :-1])/2
        # convert ndens into a list of arrays
        n_list = []
        p_list = []
        for i in range(len(t)):
            n_list.append(n_dens[i])
            p_list.append(p_dens[i])

        return n_list, p_list
    else:
        # return the full density arrays
        return n_dens, p_dens

def DBTD_multi_trap(parameters, t, Gpulse, t_span, N0=0, G_frac = 1, equilibrate=True, eq_limit=1e-2, maxcount=1e3, output_integrated_values = True,**kwargs):
    """Solve the bimolecular, multi-trap trapping and detrapping model including diffusion.  
    This implementation is based on the work by [M. Simmonds](https://github.com/MaximSimmonds-HZB/MAPI-FAPI-fitting).  

    Note: This needs to be tested further, I think there are some issues regarding the stability of the solver and whether the diffusion part was implemented properly.

    Parameters
    ----------
    parameters : dict
        dictionary containing the parameters of the model it must contain 'k_trap', 'k_direct', 'k_detrap', 'N_t_bulk' and 'N_A'.

            k_direct : float
                Bimolecular/direct recombination rate constant in m^3 s^-1
            N_t_bulk : float
                Bulk trap density (can be multiple) in m^-3
            C_n : float
                Electron capture coefficient (can be multiple) in m^3 s^-1
            C_p : float
                Hole capture coefficient (can be multiple) in m^3 s^-1
            E_t_bulk : float
                Relative trap depth in the bandgap (can be multiple) in eV
            L : float
                Length of the device in m
            alpha : float
                Absorption coefficient in m^-1
            mu_n : float
                Electron mobility in m^2 V^-1 s^-1
            mu_p : float
                Hole mobility in m^2 V^-1 s^-1
            N_c : float
                Effective density of states of the conduction band in m^-3
            N_v : float
                Effective density of states of the valence band in m^-3
            Eg : float
                Bandgap energy in eV
            T : float
                Temperature in K

    t : ndarray of shape (n,)
        time values in s
    Gpulse : ndarray of shape (n,)
        array of values of the charge carrier generation rate m^-3 s^-1
    t_span : ndarray of shape (n,), optional
        time values for the pulse time step in case it is different from t, by default None
    N0 : float, optional
        initial values of the electron, trapped electron and hole concentrations in m^-3, by default 0
    G_frac : float, optional
        fraction of the generation rate that is absorbed, by default 1
    equilibrate : bool, optional
        whether to equilibrate the system, by default True
    eq_limit : float, optional
        limit for the relative change of the last time point to the previous one to consider the system in equilibrium, by default 1e-2
    maxcount : int, optional
        maximum number of iterations to reach equilibrium, by default 1e3
    output_integrated_values : bool, optional
        whether to output the integrated values, by default True
    kwargs : dict
        additional keyword arguments for the solver function

            'method' : str, optional
                method to use for the solver, by default 'Radau'
            'rtol' : float, optional
                relative tolerance, by default 1e-3
            'atol' : float, optional
                absolute tolerance, by default 1e-6
            'grid_size' : int, optional
                size of the grid for the spatial discretization, by default 100
            'dimensionless' : bool, optional
                whether to use dimensionless variables, by default False
            'timeout' : float, optional
                maximum time to wait for the solver to finish, by default 90
            'timeout_solve' : float, optional
                maximum time to wait for solve_ivp to finish, by default 90
            'use_jacobian' : bool, optional
                whether to use the Jacobian, by default True
            
    Returns
    -------
    list or ndarray
        The integrated values of the electron density versus time and space.
        Each element of the list corresponds to a specific time point and contains the electron density values at different spatial positions.
    list or ndarray
        The integrated values of the hole density versus time and space.
        Each element of the list corresponds to a specific time point and contains the hole density values at different spatial positions.

    Raises
    ------
    ValueError
        If the parameters are not valid.

    """    
    #Extracting parameters

    pnames = [p for p in parameters.keys()]

    if 'k_direct' in pnames:
        k_direct = parameters['k_direct']
    else:
        raise ValueError('k_direct is not in the parameters dictionary')
    
    if 'L' in pnames:
        L = parameters['L']
    else:
        raise ValueError('L is not in the parameters dictionary')
    
    if 'alpha' in pnames:
        alpha = parameters['alpha']
    else:
        raise ValueError('alpha is not in the parameters dictionary')

    
    # traps 
    trapsnames = [p for p in pnames if 'N_t_bulk' in p]
    Cnnames = [p for p in pnames if 'C_n' in p]
    Cpnames = [p for p in pnames if 'C_p' in p]
    Etrapnames = [p for p in pnames if 'E_t_bulk' in p]

    # check if we are specifying the ratio instead of values of C_ps
    if len(Cpnames) != len(Cnnames):
        for p in pnames:
            if 'ratio_Cnp' in p: # adds C_p names based on the ratio
                Cpnames.append('C_p_'+p.split('_')[-1])

    if len(trapsnames) == 0 or len(Etrapnames) == 0 or len(Cnnames) == 0 or len(Cpnames) == 0 or len(trapsnames) != len(Cnnames) or len(trapsnames) != len(Cpnames) or len(trapsnames) != len(Etrapnames):
        raise ValueError('The parameters dictionary must contain at least one trap with its corresponding C_n, C_p and E_trap values')
    # check that trapnames are written in a format like 'N_t_bulk_1'
    for name,n1,n2,n3 in zip(trapsnames,Cnnames,Cpnames,Etrapnames):
        if not re.match(r'N_t_bulk_\d+', name):
            raise ValueError(f'{name} is not a valid trap name')
        if not re.match(r'C_n_\d+', n1):
            raise ValueError(f'{n1} is not a valid C_n name')
        if not re.match(r'C_p_\d+', n2):
            raise ValueError(f'{n2} is not a valid C_p name')
        if not re.match(r'E_t_bulk_\d+', n3):
            raise ValueError(f'{n3} is not a valid E_t_bulk name')

    # then reorder the lists to make sure the correct values are match together in the following
    for i in range(len(trapsnames)):
        trapsnames[i] = f'N_t_bulk_{i+1}'
        Cnnames[i] = f'C_n_{i+1}'
        Cpnames[i] = f'C_p_{i+1}'
        Etrapnames[i] = f'E_t_bulk_{i+1}'

    N_t_bulk_list, C_n_bulk_list, C_p_bulk_list, E_t_bulk_list = [], [], [], []
    for i in range(len(trapsnames)):
        N_t_bulk_list.append(parameters[trapsnames[i]])
        E_t_bulk_list.append(parameters[Etrapnames[i]])
        C_n_bulk_list.append(parameters[Cnnames[i]])
        if Cpnames[i] not in parameters.keys(): # check if we are specifying the ratio instead of values of C_ps
            C_p_bulk_list.append(parameters[Cnnames[i]]/parameters['ratio_Cnp_'+str(i+1)])
        else:
            C_p_bulk_list.append(parameters[Cpnames[i]])
    # convert as array
    N_t_bulk_list = np.asarray(N_t_bulk_list)
    C_n_bulk_list = np.asarray(C_n_bulk_list)
    C_p_bulk_list = np.asarray(C_p_bulk_list)
    E_t_bulk_list = np.asarray(E_t_bulk_list)

    if 'mu_n' in pnames and 'mu_p' in pnames:
        mu_n = parameters['mu_n']
        mu_p = parameters['mu_p']
    elif 'mu' in pnames:
        mu_n = parameters['mu']
        mu_p = parameters['mu']
    else:
        raise ValueError('mu_n and mu_p or mu must be in the parameters dictionary')
    
    if 'C_auger_n' in pnames and 'C_auger_p' in pnames:
        C_auger_n = parameters['C_auger_n']
        C_auger_p = parameters['C_auger_p']
    elif 'C_auger' in pnames:
        C_auger_n = parameters['C_auger_n']
        C_auger_p = parameters['C_auger_p']
    else:
        C_auger_n = C_auger_p = 0.0

    if 'N_c' in pnames and 'N_v' in pnames:
        N_c = parameters['N_c']
        N_v = parameters['N_v']
    elif 'N_cv' in pnames:
        N_c = parameters['N_cv']
        N_v = parameters['N_cv']
    else:
        raise ValueError('N_c and N_v or N_cv must be in the parameters dictionary')
    
    if 'Eg' in pnames:
        Eg = parameters['Eg']
    else:
        raise ValueError('Eg must be in the parameters dictionary')
    
    if 'T' in pnames:
        T = parameters['T']
    else:
        T = 300
    
    # kwargs
    dimensionless = kwargs.get('dimensionless', True)
    grid_size = kwargs.get('grid_size', 100)  # number of grid points
    timeout = kwargs.get('timeout', 60)
    timeout_solve = kwargs.get('timeout_solve', 60)
    method = kwargs.get('method', 'BDF')  # default method for solve_ivp
    use_jacobian = kwargs.get('use_jacobian', True)
    if method == 'LSODA':
        use_jacobian = False
    rtol = kwargs.get('rtol', 1e-3)
    atol = kwargs.get('atol', 1e-6)
    
    # Derived quantities
    ni = np.sqrt(N_c*N_v*np.exp(-Eg/(kb*T))) # intrinsic carrier concentration in m^-3
    p1s = N_v*np.exp(-E_t_bulk_list/(kb*T)) 
    n1s = N_c*np.exp((E_t_bulk_list-Eg)/(kb*T))  
    D_n = mu_n * kb * T  # electron diffusion coefficient in m^2 s^-1
    D_p = mu_p * kb * T  # hole diffusion coefficient in m^2 s^-1
    
    ft = (C_n_bulk_list*ni + C_p_bulk_list*p1s)/(C_n_bulk_list *(ni + n1s) + C_p_bulk_list*(p1s + ni)) # proportion of electrons that are trapped (filling probability at steady state, in the dark)

    number_of_traps = len(N_t_bulk_list)
    z_array = np.linspace(0, L, grid_size)  # spatial domain
    dz = z_array[1] - z_array[0]  # spatial step size

    mean_beer_lambert = np.mean(np.exp(-alpha * z_array))  # Beer-Lambert law for generation profile
    generation = np.zeros((len(t_span), len(z_array)))
    for i in range(len(t_span)):
        generation[i] = Gpulse[i] * np.exp(-alpha * z_array) / mean_beer_lambert  # normalize the generation profile

    N_init = N0 * G_frac# initial conditions
    n0_z = ((N_init * L)/np.trapezoid(np.exp(-alpha * z_array), x=z_array)) * np.exp(-alpha * z_array)
    
    ## Initial population distribution (uniform for simplicity)
    ## Compute the absorption profile for n (We consider that the n and p diffusion constants are similar... maybe I souldn't), consider constant distribution for the trapped charges at equilibria. 
    ## For now, its constant constant.
    P_init = np.zeros((len(N_t_bulk_list)+2, grid_size))
    for j in range(len(P_init)):
        if (j<2):
            P_init[j, :] = n0_z
        else:
            P_init[j, :] = N_t_bulk_list[j-2]*ft[j-2]
    
    # Flatten the initial conditions into a single vector
    P0 = P_init.flatten()
    arg = [k_direct, Eg, N_t_bulk_list, C_n_bulk_list, C_p_bulk_list, E_t_bulk_list, N_c, N_v, T, D_n, D_p, C_auger_n, C_auger_p, number_of_traps, grid_size, dz]

    # Compute the second derivative of the populations
    def second_derivative(P, N, dz):
        d2P = np.zeros_like(P)
        d2P[1:-1] = (P[2:] - 2 * P[1:-1] + P[:-2]) / dz**2
        # Zero-flux boundary conditions (d/dx = 0 at boundaries)
        ## M.S Correction: factor 2 in front of both boundaries, which is the correct boundary condition
        d2P[0] = 2*(P[1] - P[0]) / (dz ** 2)  # Left boundary (forward difference)
        d2P[-1] = 2*(P[-2] - P[-1]) / (dz ** 2)  # Right boundary (backward difference)
        return d2P

    def model_vect(t, P_flat, kdirect, Eg, Bulk_tr, Bn, Bp, ETrap, Nc, Nv, T, D_n, D_p, C_auger_n, C_auger_p, number_of_traps, grid_size, dz):
        if P_flat.ndim == 1:
            P_flat = P_flat[:, None]  # make it (n_variables, 1)

        n_times = P_flat.shape[1]
        
        P = P_flat.reshape(number_of_traps + 2, grid_size, n_times)  # Now (populations, space, n_times)
        n = P[0]  # shape (space, n_times)
        p = P[1]
        ntr = P[2:]

        kT = kb * T
        ni2 = Nc*Nv*np.exp(-Eg/kT)
        ni2 = ni2 * np.ones((grid_size, n_times))

        # Vectorized capture/emission
        e_capture = Bn[:, None, None] * n[None, :, :] * (Bulk_tr[:, None, None] - ntr)
        h_capture = Bp[:, None, None] * p[None, :, :] * ntr
        e_emission = (Nc * np.exp(-(Eg - ETrap) / kT) * Bn)[:, None, None] * ntr
        h_emission = (Nv * np.exp(-ETrap / kT) * Bp)[:, None, None] * (Bulk_tr[:, None, None] - ntr)

        # Diffusion terms
        d2n = np.array([second_derivative(n[:, i], grid_size, dz) for i in range(n_times)]).T
        d2p = np.array([second_derivative(p[:, i], grid_size, dz) for i in range(n_times)]).T

        dPdt = np.zeros_like(P)

        dPdt[0] = - kdirect * (n * p - ni2) - np.sum(e_capture, axis=0) + np.sum(e_emission, axis=0) + D_n * d2n - (C_auger_n * n * n *p + C_auger_p* n*p*p)
        dPdt[1] = - kdirect * (n * p - ni2) - np.sum(h_capture, axis=0) + np.sum(h_emission, axis=0) + D_p * d2p - (C_auger_n * n * n *p + C_auger_p* n*p*p)

        for i in range(number_of_traps):
            dPdt[i+2] = e_capture[i] - e_emission[i] - h_capture[i] + h_emission[i]

        return dPdt.reshape(-1, n_times)
    
    def jacobian_no_flux_vectorized_fixed(t, P_flat, *args):

        kdirect, Eg, Bulk_tr, Bn, Bp, ETrap, Nc, Nv, T, D_n, D_p, C_auger_n, C_auger_p, number_of_traps, grid_size, dz = args

        N_pop = number_of_traps + 2
        nvars = N_pop * grid_size
        #print("time", t)
        P = P_flat.reshape(N_pop, grid_size)
        n = P[0]
        p = P[1]
        ntr = P[2:]

        kT = kb * T

        exp_e = Nc * np.exp(-(Eg - ETrap) / kT)
        exp_h = Nv * np.exp(-(ETrap) / kT)

        sum_Bn_Bulktr_ntr = np.sum(Bn[:, None] * (Bulk_tr[:, None] - ntr), axis=0)
        sum_Bp_ntr = np.sum(Bp[:, None] * ntr, axis=0)

        idx_n = np.arange(grid_size)
        idx_p = grid_size + np.arange(grid_size)
        idx_traps = [grid_size * (2 + j) + np.arange(grid_size) for j in range(number_of_traps)]

        diag_n = -kdirect * p - sum_Bn_Bulktr_ntr - (C_auger_n * 2*n * p + C_auger_p * p * p)
        diag_p = -kdirect * n - sum_Bp_ntr - (C_auger_n * n * n + C_auger_p * 2*p * n)

        J = lil_matrix((nvars, nvars))

        # Diffusion for n
        diag_n[1:-1] += -2 * D_n / dz**2
        J[idx_n[1:-1], idx_n[:-2]] = D_n / dz**2
        J[idx_n[1:-1], idx_n[2:]]  = D_n / dz**2
        diag_n[0] += -2 * D_n / dz**2
        J[idx_n[0], idx_n[1]] = 2 * D_n / dz**2
        diag_n[-1] += -2 * D_n / dz**2
        J[idx_n[-1], idx_n[-2]] = 2 * D_n / dz**2

        # Diffusion for p
        diag_p[1:-1] += -2 * D_p / dz**2
        J[idx_p[1:-1], idx_p[:-2]] = D_p / dz**2
        J[idx_p[1:-1], idx_p[2:]]  = D_p / dz**2
        diag_p[0] += -2 * D_p / dz**2
        J[idx_p[0], idx_p[1]] = 2 * D_p / dz**2
        diag_p[-1] += -2 * D_p / dz**2
        J[idx_p[-1], idx_p[-2]] = 2 * D_p / dz**2

        J[idx_n, idx_n] = diag_n
        J[idx_p, idx_p] = diag_p

        # Radiative and Auger terms
        J[idx_n, idx_p] = -kdirect * n - (C_auger_n * n * n + C_auger_p * 2*p * n) 
        J[idx_p, idx_n] = -kdirect * p - (C_auger_n * 2*n * p + C_auger_p * p * p)

        for j in range(number_of_traps):
            idx_trap = idx_traps[j]
            # Assign trap diagonal block (important!), sign change for the hole emission to (-) and not (+). Does not affect the final decays, just makes the solver struggle in the regions where this term is important.
            J[idx_trap, idx_trap] = -Bn[j] * n - Bn[j] * exp_e[j] - Bp[j] * p - Bp[j] * exp_h[j]

            # Trap eqns derivatives w.r.t n and p (diagonal only)
            J[idx_trap, idx_n] = Bn[j] * (Bulk_tr[j] - ntr[j])
            J[idx_trap, idx_p] = -Bp[j] * ntr[j]

            # Electron and hole eqns derivatives w.r.t trap populations (diagonal only)
            J[idx_n, idx_trap] = Bn[j] * n + Bn[j] * exp_e[j]
            J[idx_p, idx_trap] = -Bp[j] * p - Bp[j] * exp_h[j]
        
        return J.tocsc()

    def model_vect_dimensionless(t, P_flat_d, kdirect, Eg, Bulk_tr_d, Bn, Bp, ETrap, Nc_d, Nv_d, T, 
                                    D_n, D_p, C_auger_n, C_auger_p, number_of_traps, grid_size, dz):    
        if P_flat_d.ndim == 1:
            P_flat_d = P_flat_d[:, None]  # make it (n_variables, 1)

        n_times = P_flat_d.shape[1]
        
        P = P_flat_d.reshape(number_of_traps + 2, grid_size, n_times)  # Now (populations, space, n_times)
        n_d = P[0]  # shape (space, n_times)
        p_d = P[1]
        ntr_d = P[2:]

        kT = kb * T
        
        # Vectorized capture/emission
        e_capture = Bn[:, None, None] * n_d[None, :, :] * (Bulk_tr_d[:, None, None] - ntr_d)
        h_capture = Bp[:, None, None] * p_d[None, :, :] * ntr_d
        e_emission = (Nc_d * np.exp(-(Eg - ETrap) / kT) * Bn)[:, None, None] * ntr_d
        h_emission = (Nv_d * np.exp(-ETrap / kT) * Bp)[:, None, None] * (Bulk_tr_d[:, None, None] - ntr_d)

        # Diffusion terms
        d2n = np.array([second_derivative(n_d[:, i], grid_size, dz) for i in range(n_times)]).T
        d2p = np.array([second_derivative(p_d[:, i], grid_size, dz) for i in range(n_times)]).T

        dPdt = np.zeros_like(P)
        dPdt[0] = - kdirect * (n_d * p_d - 1) - np.sum(e_capture, axis=0) + np.sum(e_emission, axis=0) + D_n * d2n - (C_auger_n * n_d * n_d * p_d + C_auger_p * n_d * p_d * p_d)
        dPdt[1] = - kdirect * (n_d * p_d - 1) - np.sum(h_capture, axis=0) + np.sum(h_emission, axis=0) + D_p * d2p - (C_auger_n * n_d * n_d * p_d + C_auger_p * n_d * p_d * p_d)

        for i in range(number_of_traps):
            dPdt[i+2] = (e_capture[i] - e_emission[i] - h_capture[i] + h_emission[i])

        return dPdt.reshape(-1, n_times)
        
    RealChange,diff = 1e40,1e40 # artificially large to start with
    end_point = 1e-20
    if dimensionless:
        n0_z = n0_z / ni  # non-dimensionalize the initial charge carrier density
        P0 = P0 / ni
        x = np.linspace(0, 1, grid_size)
        dx = x[1] - x[0]
        taus = 1/(N_t_bulk_list * np.sqrt(C_n_bulk_list * C_p_bulk_list))
        taus = taus[~np.isinf(taus)] # remove from the array the inf
        tau = np.average(taus)

        t_span = t_span / tau
        D_n = D_n * tau/(L**2)
        D_p = D_p * tau/(L**2)
        generation = generation * tau / ni

        arg = [k_direct * ni * tau, Eg, N_t_bulk_list/ni, C_n_bulk_list * ni * tau, C_p_bulk_list * ni * tau, E_t_bulk_list, N_c/ni, N_v/ni, T, D_n, D_p, C_auger_n*(ni**2)*tau, C_auger_p*(ni**2)*tau, number_of_traps, grid_size, dx]
    
    t_start = time.time()
    count = 0
    
    try:
        if equilibrate:
            while True:
                # print(f"Equilibrating {count} times ",parameters, 'realChange',np.mean(RealChange))
                # print(time.time()- t_start, np.mean(RealChange))
                start_time = time.time()
                def timeout_event(*args):
                    return min(time.time() - start_time - timeout_solve, 0)  # zero when runtime > timeout
                timeout_event.terminal = True  # stop integration
                timeout_event.direction = 1

                if time.time() - t_start > timeout:
                    logger.warning(f"Equilibration took too long, stopping. RealChange mean: {np.mean(RealChange)}, Parameters: {parameters}")

                    return np.nan * np.ones((len(t),grid_size)), np.nan * np.ones((len(t),grid_size))

                if dimensionless:
                    if use_jacobian:
                        sol_single = solve_ivp(model_vect_dimensionless, [t_span[0],t_span[-1]], P0, method=method, args=arg, vectorized=True,  rtol=rtol, atol=atol,t_eval=t_span,jac=jacobian_no_flux_vectorized_fixed, events=timeout_event)
                    else:
                        sol_single = solve_ivp(model_vect_dimensionless, [t_span[0],t_span[-1]], P0, method=method, args=arg, vectorized=True,  rtol=rtol, atol=atol,t_eval=t_span, events=timeout_event)
                else:
                    if use_jacobian:
                        sol_single = solve_ivp(model_vect, [t_span[0],t_span[-1]], P0, method=method, args=arg, vectorized=True,  rtol=rtol, atol=atol,t_eval=t_span,jac=jacobian_no_flux_vectorized_fixed, events=timeout_event)
                    else:
                        sol_single = solve_ivp(model_vect, [t_span[0],t_span[-1]], P0, method=method, args=arg, vectorized=True,  rtol=rtol, atol=atol,t_eval=t_span, events=timeout_event)

                if not(sol_single.success):
                    logger.warning("ODE solver did not converge, returning NaN arrays.")
                    return np.nan * np.ones((len(t),grid_size)), np.nan * np.ones((len(t),grid_size))

                sol_flat = sol_single.y.reshape(len(P_init), grid_size, -1)
                n_last = sol_flat[0, :, -1]
                p_last = sol_flat[1, :, -1]
                # Inject fresh carriers
                n_next = n_last + n0_z
                p_next = p_last + n0_z
                # Update initial condition for next iteration
                P0[:grid_size] = n_next
                P0[grid_size:2*grid_size] = p_next
                for j in range(len(N_t_bulk_list)):
                    u = j+2
                    P0[u*grid_size:(u+1)*grid_size] = sol_flat[u,:,-1]

                new_end = n_last#(sol_flat[:,0,-1])
                RealChange  = abs((new_end - end_point)/end_point) # relative change of mean
                end_point = new_end
                count += 1
                # print(f'Equilibrating: {count} iterations, Relative Change: {np.max(abs(RealChange))}', parameters)
                # if np.mean(RealChange) < eq_limit and not np.all(abs(RealChange) < eq_limit):
                #     # count number of false in np.all(abs(RealChange)
                #     false_count = np.sum(~np.all(abs(RealChange) < eq_limit))
                #     print(false_count, 'points not yet converged in count', count)
                if np.all(abs(RealChange) < eq_limit) or count > maxcount or np.sum(diff == 0):
                    if count > maxcount:
                        logger.warning(f"Equilibration did not converge within the maximum number of iterations.")
                        return np.nan * np.ones((len(t),grid_size)), np.nan * np.ones((len(t),grid_size))
                    
                    break
        

        # Now run the simulation with the right time
        # print("Running the simulation ad",parameters)
        if dimensionless:
            t = t/ tau
            if use_jacobian:
                sol = solve_ivp(model_vect_dimensionless, [t[0], t[-1]], P0, method=method, args=arg, vectorized=True, t_eval=t, rtol=rtol, atol=atol,jac=jacobian_no_flux_vectorized_fixed)
            else:
                sol = solve_ivp(model_vect_dimensionless, [t[0], t[-1]], P0, method=method, args=arg, vectorized=True, t_eval=t, rtol=rtol, atol=atol)
        else:
            if use_jacobian:
                sol = solve_ivp(model_vect, [t[0], t[-1]], P0, method=method, args=arg, vectorized=True, t_eval=t, rtol=rtol, atol=atol,jac=jacobian_no_flux_vectorized_fixed)
            else:
                sol = solve_ivp(model_vect, [t[0], t[-1]], P0, method=method, args=arg, vectorized=True, t_eval=t, rtol=rtol, atol=atol)
        # print('done',parameters)
        # if dimensionless:
        #     t = t/ tau
        #     if use_jacobian:
        #         sol = solve_ivp(model_vect_dimensionless, [t_span[0], t_span[-1]], P0, method=method, args=arg, vectorized=True, t_eval=t, rtol=rtol, atol=atol,jac=jacobian_no_flux_vectorized_fixed)
        #     else:
        #         sol = solve_ivp(model_vect_dimensionless, [t_span[0], t_span[-1]], P0, method=method, args=arg, vectorized=True, t_eval=t, rtol=rtol, atol=atol)
        # else:
        #     if use_jacobian:
        #         sol = solve_ivp(model_vect, [t_span[0], t_span[-1]], P0, method=method, args=arg, vectorized=True, t_eval=t, rtol=rtol, atol=atol,jac=jacobian_no_flux_vectorized_fixed)
        #     else:
        #         sol = solve_ivp(model_vect, [t_span[0], t_span[-1]], P0, method=method, args=arg, vectorized=True, t_eval=t, rtol=rtol, atol=atol)
        if not(sol.success):
            logger.warning("ODE solver did not converge, returning NaN arrays.")
            return np.nan * np.ones((len(t),grid_size)), np.nan * np.ones((len(t),grid_size))
        
        sol_flat = sol.y.reshape(len(P_init), grid_size, -1)
        n_dens = sol_flat[0, :, :].T  # electron density
        p_dens = sol_flat[1, :, :].T  # hole density
    
        if dimensionless:
            n_dens = n_dens * ni
            p_dens = p_dens * ni

        if output_integrated_values:
            n_dens = (n_dens[:, 1:]+ n_dens[:, :-1])/2
            p_dens = (p_dens[:, 1:]+ p_dens[:, :-1])/2

            # convert ndens into a list of arrays
            n_list = []
            p_list = []
            for i in range(len(t)):
                n_list.append(n_dens[i])
                p_list.append(p_dens[i])

            return n_list, p_list
        else:
            return n_dens, p_dens
    except Exception as e:
        print(e)
        print("An error occurred during the simulation: {}".format(e))
        return np.nan * np.ones((len(t),grid_size)), np.nan * np.ones((len(t),grid_size))

