"""Rate equation models for charge carrier dynamics in semiconductors"""
# Note: This class is inspired by the https://github.com/i-MEET/boar/ package
######### Package Imports #########################################################################

import warnings
import numpy as np
from scipy.integrate import solve_ivp, odeint
from functools import partial

######### Function Definitions ####################################################################
def BT_model(ktrap, k_direct, t, Gpulse, tpulse, ninit=[0],  equilibrate=True, eq_limit=1e-2, maxcount=1e3, solver_func = 'solve_ivp', **kwargs):
    """Solve the bimolecular trapping equation :  
    
    dn/dt = G - ktrap * n - k_direct * n^2
   

    Parameters
    ----------
    t : ndarray of shape (n,)
        array of time values

    G :  ndarray of shape (n,)
        array of values of the charge carrier generation rate m^-3 s^-1

    ktrap : float
        trapping rate constant

    k_direct : float
        Bimolecular/direct recombination rate constant

    tpulse : ndarray of shape (n,), optional
        array of time values for the pulse time step in case it is different from t, by default None

    ninit : list of floats, optional
        initial value of the charge carrier density, by default [0]
    
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
    # check solver function
    if solver_func not in ['odeint','solve_ivp']:
        warnings.warn('solver function not recognized, using odeint', UserWarning)
        solver_func = 'odeint'

    # kwargs
    method = kwargs.get('method', 'RK45')
    rtol = kwargs.get('rtol', 1e-6)

    # check if the pulse time step is different from the time vector
    if tpulse is None:
        tpulse = t

    def dndt(t, y, tpulse, Gpulse, ktrap, k_direct):
        """Bimolecular trapping equation
        """  
        gen = np.interp(t, tpulse, Gpulse) # interpolate the generation rate at the current time point
        
        S = gen - ktrap * y - k_direct * y**2
        return S.T

    # Solve the ODE
    if equilibrate: # make sure the system is in equilibrium 
        # to be sure we equilibrate the system properly we need to solve the dynamic equation over the full range of 1/fpu in time
        rend = 1e-20 # last time point
        RealChange = 1e19 # initialize the relative change with a high number
        rstart = ninit[0]+rend
        count = 0
        while np.any(abs(RealChange) > eq_limit) and count < maxcount:
            if solver_func == 'odeint':
                r = odeint(dndt, rstart, tpulse, args=(tpulse, Gpulse, ktrap, k_direct), tfirst=True, **kwargs)
                RealChange = (r[-1] -rend)/rend # relative change of mean
                rend = r[-1] # last time point
            elif solver_func == 'solve_ivp':
                # r = solve_ivp(dndt, [t[0], t[-1]], rstart, args=(tpulse, Gpulse, ktrap, k_direct), method = method, rtol=rtol)
                r = solve_ivp(partial(dndt,tpulse = tpulse, Gpulse = Gpulse, ktrap = ktrap, k_direct = k_direct), [t[0], t[-1]], ninit, t_eval = t, method = method, rtol=rtol)
    
                RealChange  = (r.y[:,-1] -rend)/rend # relative change of mean
                rend = r.y[:,-1] # last time point
            rstart = ninit[0]+rend
            count += 1

    else:
        rstart = ninit[0]
    
    # solve the ODE again with the new initial conditions with the equilibrated system and the original time vector
    Gpulse_eq = np.interp(t, tpulse, Gpulse) # interpolate the generation rate at the current time point
    if solver_func == 'odeint':
        r = odeint(dndt, rstart, t, args=(t, Gpulse_eq, ktrap, k_direct), tfirst=True, **kwargs)
        return r[:,0]
    elif solver_func == 'solve_ivp':
        # r = solve_ivp(dndt, [t[0], t[-1]], rstart, t_eval = t, args=(t, Gpulse_eq, ktrap, k_direct), method = method, rtol=rtol)
        r = solve_ivp(partial(dndt,tpulse = t, Gpulse = Gpulse_eq, ktrap = ktrap, k_direct = k_direct), [t[0], t[-1]], rend + ninit[0], t_eval = t, method = method, rtol=rtol)
    
        return r.y[0]


def BTD_model(ktrap, k_direct, kdetrap, N_t_bulk, p_0, t, Gpulse, tpulse, ninit=[0,0,0], equilibrate=True, eq_limit=1e-2,maxcount=1e3, solver_func = 'odeint',**kwargs):
    """Solve the bimolecular trapping and detrapping equation :

    dn/dt = G - ktrap * n * (N_t_bulk - n_t) - k_direct * n * (p + p_0)
    dn_t/dt = k_trap * n * (N_t_bulk - n_t) - kdetrap * n_t * (p + p_0)
    dp/dt = G - kdetrap * n_t * (p + p_0) - k_direct * n * (p + p_0)

    Parameters
    ----------
    ktrap : float
        Trapping rate constant in m^3 s^-1
    k_direct : float
        Bimolecular/direct recombination rate constant in m^3 s^-1
    kdetrap : float
        Detrapping rate constant in m^3 s^-1
    N_t_bulk : float
        Bulk trap density in m^-3
    p_0 : float
        Ionized p-doping concentration in m^-3
    t : ndarray of shape (n,)
        time values in s
    Gpulse : ndarray of shape (n,)
        array of values of the charge carrier generation rate m^-3 s^-1
    tpulse : ndarray of shape (n,), optional
        time values for the pulse time step in case it is different from t, by default None
    ninit : list of floats, optional
        initial electron, trapped electron and hole concentrations in m^-3, by default [0,0,0]
    equilibrate : bool, optional
        whether to equilibrate the system, by default True
    eq_limit : float, optional
        limit for the relative change of the last time point to the previous one to consider the system in equilibrium, by default 1e-2
    maxcount : int, optional
        maximum number of iterations to reach equilibrium, by default 1e3
    solver_func : str, optional
        solver function to use can be ['odeint','solve_ivp'], by default 'odeint'
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
        trapped electron concentration in m^-3
    ndarray of shape (n,)
        hole concentration in m^-3
    """   

    # check solver function
    if solver_func not in ['odeint','solve_ivp']:
        warnings.warn('solver function not recognized, using odeint', UserWarning)
        solver_func = 'odeint'

    # kwargs
    method = kwargs.get('method', 'RK45')
    rtol = kwargs.get('rtol', 1e-3)

    # check if the pulse time step is different from the time vector
    if tpulse is None:
            tpulse = t
    
    def rate_equations(t, n, tpulse, Gpulse, ktrap, k_direct, kdetrap, N_t_bulk, p_0):
            """Rate equation of the BTD model (PEARS) 

            Parameters
            ----------
            t : float
                time in s
            n : list of floats
                electron, trapped electron and hole concentrations in m^-3
            Gpulse : ndarray of shape (n,)
                array of values of the charge carrier generation rate m^-3 s^-1
            tpulse : ndarray of shape (n,), optional
                array of time values for the pulse time step in case it is different from t, by default None
            ktrap : float
                trapping rate constant in m^3 s^-1
            k_direct : float
                Bimolecular/direct recombination rate constant in m^3 s^-1
            kdetrap : float
                detrapping rate constant in m^3 s^-1
            N_t_bulk : float
                bulk trap density in m^-3
            p_0 : float
                ionized p-doping concentration in m^-3

            Returns
            -------
            list
                Fractional change of electron, trapped electron and hole concentrations at times t
            """

            gen = np.interp(t, tpulse, Gpulse) # interpolate the generation rate at the current time point
            
            n_e, n_t, n_h = n
            
            B = k_direct * n_e * (n_h + p_0)
            T = ktrap * n_e * (N_t_bulk - n_t)
            D = kdetrap * n_t * (n_h + p_0)
            dne_dt = gen - B - T
            dnt_dt = T - D
            dnh_dt = gen - B - D
            return [dne_dt, dnt_dt, dnh_dt]

    # Solve the ODE
    if equilibrate: # equilibrate the system
        # to be sure we equilibrate the system properly we need to solve the dynamic equation over the full range of 1/fpu in time 
        rend = [1e-20,1e-20,1e-20] # initial conditions
        rstart = [rend[0] + ninit[0], rend[1] + ninit[1], rend[2] + ninit[2]] # initial conditions for the next integration
        RealChange = 1e19 # initialize the relative change with a high number
        count = 0
        while np.any(abs(RealChange) > eq_limit) and count < maxcount:

            if solver_func == 'solve_ivp':
                r = solve_ivp(partial(rate_equations,tpulse = tpulse, Gpulse = Gpulse, ktrap = ktrap, k_direct = k_direct, kdetrap = kdetrap, N_t_bulk = N_t_bulk, p_0 = p_0), [t[0], t[-1]], rstart, t_eval = None, method = method, rtol= rtol) # method='LSODA','RK45'
                # monitor only the electron concentration           
                RealChange  = (r.y[0,-1] - rend[0])/rend[0] # relative change of mean
                rend = [r.y[0,-1], r.y[1,-1], r.y[2,-1]] # last time point
            elif solver_func == 'odeint':
                r = odeint(rate_equations, rstart, tpulse, args=(tpulse, Gpulse, ktrap, k_direct, kdetrap, N_t_bulk, p_0), tfirst=True, rtol=rtol)
                RealChange = (r[-1,0]-rend[0])/rend[0] # relative change of mean
                rend = [r[-1,0], r[-1,1], r[-1,2]] # last time point

            rstart = [rend[0] + ninit[0], rend[1] + ninit[1], rend[2] + ninit[2]] # initial conditions for the next integration
            count += 1
    else:
        rstart = ninit


    # solve the ODE again with the new initial conditions with the equilibrated system and the original time vector
    Gpulse_eq = np.interp(t, tpulse, Gpulse) # interpolate the generation rate at the current time point
    if solver_func == 'solve_ivp':
        r = solve_ivp(partial(rate_equations,tpulse = t, Gpulse = Gpulse_eq, ktrap = ktrap, k_direct = k_direct, kdetrap = kdetrap, N_t_bulk = N_t_bulk, p_0 = p_0), [t[0], t[-1]], rstart, t_eval = t, method = method, rtol= rtol) # method='LSODA','RK45'
        n_e = r.y[0]
        n_t = r.y[1]
        n_h = r.y[2]
    elif solver_func == 'odeint':
        r = odeint(rate_equations, rstart, t, args=(t, Gpulse_eq, ktrap, k_direct, kdetrap, N_t_bulk, p_0), tfirst=True, rtol=rtol)
        n_e = r[:,0]
        n_t = r[:,1]
        n_h = r[:,2]

    return n_e, n_t, n_h

