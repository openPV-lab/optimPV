"""Rate equation models for charge carrier dynamics in semiconductors"""
# Note: This class is inspired by the https://github.com/i-MEET/boar/ package
######### Package Imports #########################################################################

import warnings
import numpy as np
from scipy.integrate import solve_ivp, odeint
from functools import partial

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
        rstart = N0


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

