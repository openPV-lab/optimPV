"""Pump pulse generation for transient experiments"""
# Note: This class is inspired by the https://github.com/i-MEET/boar/ package
######### Package Imports #########################################################################

import warnings
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import square
from scipy import constants


## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')
c = constants.value(u'speed of light in vacuum')
h_eV = constants.value(u'Planck constant in eV s')
h_J = constants.value(u'Planck constant')

######### Function Definitions ####################################################################
def get_flux_density(Power,wavelength,fpu,Area,alpha):
    """From the measured power, repetition rate and area, 
    get photons/m2 and approximate photons/m3 per pulse

    Parameters
    ----------
    Power : float
        total CW power of pulse in W
    wavelength : float
        excitation wavelength in m
    fpu : float
        repetition rate in Hz
    Area: float
        effective pump area in m^-2
    alpha : float
        penetration depth in m

    Returns
    -------
    flux : float
        flux in photons m^-2
    density : float
        average volume density in photons m^-3
    """    
    
    E_ = h_J*c/(wavelength) # convert wavelength to J for a single photon
    Epu = Power/fpu # energy in J of a single pulse
    Nph = Epu/E_ # Photon number in pulse
    flux = Nph/Area # flux in photons m^-2
    density = flux/alpha # average absorbed density in photons m^-3
    return flux, density
    
    
    
def square_pump(t, fpu, pulse_width, P, t0 = 0, background=0,G_frac=1):
    """Square pump pulse

    Parameters
    ----------
    t : ndarray of shape (n,)
        array of time values in seconds

    fpu : float
        pump frequency in Hz

    pulse_width : float
        width of the pump pulse in seconds

    P : float
        total volume density of generated photons m^-3
    
    t0 : float, optional
        time shift of the pump pulse, by default 0
    
    background : float, optional
        background volume density of generated photons, by default 0
    
    G_frac : float, optional
        scaling for the power of the pulse, by default 1

    Returns
    -------
    ndarray of shape (n,)
        density of generated photons m^-3 at each time point

    """ 

    #convert pulse_width to fraction of the period
    pulse_width = pulse_width / (1/fpu) 
    
    pump = 0.5*square(2 * np.pi * fpu * (t-t0), pulse_width) + 0.5 # pump pulse
    putot = trapezoid(pump,t) # total pump power
    pump = pump / putot * P * G_frac # normalize the pump pulse to the total pump power
    pump = pump + background # add background

    return pump

def gaussian_pulse_norm(t, tpulse, width):
    """Returns a gaussian pulse

    Parameters
    ----------
    t : 1-D sequence of floats
        t time axis (unit: s)
    tpulse : float
        tpulse center of the pulse (unit: s)
    width : float
        width of the pulse (unit: s)

    Returns
    -------
    1-D sequence of floats
        Vector containing the gaussian pulse
    """    
    return np.exp(-np.power(t - tpulse, 2.) / (2 * np.power(width, 2.)))

def gaussian_pump(t, fpu,  pulse_width, P, t0, background=0, G_frac=1):
    """Gaussian pump pulse

    Parameters
    ----------
    t : ndarray of shape (n,)
        array of time values in seconds
    
    fpu : float
        pump frequency in Hz
    
    pulse_width : float
        width of the pump pulse in seconds
    
    P : float
        total volume density of generated photons m^-3
    
    t0 : float
        center of the pulse in seconds

    background : float, optional
        background volume density of generated photons, by default 0

    G_frac : float, optional
        scaling for the power of the pulse, by default 1

    Returns
    -------
    ndarray of shape (n,)
        density of generated photons m^-3 at each time point
    """  
    # find time step in t
    max_dt = 0
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]
        if dt > max_dt:
            max_dt = dt
    # check if the pulse is smaller than the time step
    if pulse_width < max_dt:
        # raise ValueError('The pulse width is smaller than the time step. Increase the pulse width or decrease the time step.')
        # raise warning if the pulse is smaller than the time step
        warnings.warn('The pulse width is smaller than the max time step. If you are using s linear time step you need to increase the pulse width or decrease the time step. If you are using non-linear time step make sure that you have small enough time step around the pulse or some pulses might not appear.')

    if max(t) > 1/fpu:
        # number of pulses
        Np = int(max(t) * fpu)
        # time axis for the pulses
        tp = np.linspace(0, 1/fpu, int(1e4))
        # pump pulse
        pp = gaussian_pulse_norm(tp, 0.5*1/fpu +t0, pulse_width)
        # total pump power
        putot = trapezoid(pp,tp)
        # normalize the pump pulse to the total pump power
        pp = pp / putot * P * G_frac
        # add the pulses
        for i in range(Np+1):
            if i == 0:
                pump = np.interp(t + 0.5*1/fpu , tp + i*1/fpu , pp) 
            else: 
                pump = pump + np.interp(t + 0.5*1/fpu , tp + i*1/fpu , pp) 
    else:

        pump = gaussian_pulse_norm(t , t0, pulse_width) # pump pulse

        putot = trapezoid(pump,t) # total pump power

        pump = pump / putot * P * G_frac# normalize the pump pulse to the total pump power

     


    pump = pump + background # add background

    return pump

def initial_carrier_density(t, fpu, N0, background = 0, G_frac = 1):
    """Initial carrier density

    Parameters
    ----------
    t : ndarray of shape (n,)
        array of time values in seconds
    
    fpu : float
        pump frequency in Hz
    
    N0 : float
        initial carrier density in m^-3
    
    background : float, optional
        background carrier density, by default 0

    G_frac : float, optional
        scaling for the power of the pulse, by default 1    
    

    Returns
    -------
    ndarray of shape (n,)
        initial carrier density in m^-3
    """   
    n = np.zeros(len(t))
    # repeat the initial carrier density every 1/fpu
    count = 1
    # n[0] = N0*G_frac # was like this but it is actually redondant with the fact that N0 is passed to the solver directly in RateEqModel
    n[0] = 0
    for idx, tt in enumerate(t):
        if idx == 0:
            continue

        if tt >= count/fpu and t[idx-1] <= count/fpu:
            n[idx] = N0 * G_frac
            count += 1
    

    n = n + background 
    return n
