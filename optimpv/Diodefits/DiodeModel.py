import numpy as np
from scipy.special import lambertw
from scipy import constants

## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')

def NonIdealDiode_dark(V, J0, n, R_series, R_shunt, T = 300):
    """ Solve non ideal diode equation for dark current
        J = J0*[exp(-(V-J*R_series)/(n*Vt*)) - 1] + (V - J*R_series)/R_shunt
        with the method described in:
        Solid-State Electronics 44 (2000) 1861-1864,
        see equation (4)-(5)
    

    Parameters
    ----------
    V : 1-D sequence of floats
        Array containing the voltages.
    J0 : float
        Dark Saturation Current.
    n : float
        Ideality factor.
    R_series : float
        Series resistance.
    R_shunt : float
        Shunt resistance.
    T : float, optional
        Absolute temperature , by default 300

    Returns
    -------
    1-D sequence of floats
        Array containing the currents.
    """    
    Vt = kb*T
    w = lambertw(((J0*R_series*R_shunt)/(n*Vt*(R_series+R_shunt)))*np.exp((R_shunt*(V+J0*R_series))/(n*Vt*(R_series+R_shunt)))) # check equation (5) in the paper

    Current = (n*Vt/R_series) * w + ((V-J0*R_shunt)/(R_series+R_shunt))
    return Current.real

def NonIdealDiode_light(V,J0,n,R_series,R_shunt,Jph,T=300):
    """ Solve non ideal diode equation for light current
        J = Jph - J0*[exp(-(V-J*R_series)/(n*Vt*)) - 1] - (V - J*R_series)/R_shunt
        with the method described in:
        Solar Energy Materials & Solar Cells 81 (2004) 269-277
        see equation (1)-(2)


    Parameters
    ----------
    V : 1-D sequence of floats
        Array containing the voltages.
    J0 : float
        Dark Saturation Current.
    n : float
        Ideality factor.
    R_series : float
        Series resistance.
    R_shunt : float
        Shunt resistance.
    Jph : float
        Photocurrent.
    T : float, optional
        Absolute temperature , by default 300

    Returns
    -------
    1-D sequence of floats
        Array containing the currents.
    """   

    Vt = kb*T
    w = lambertw(((J0*R_series*R_shunt)/(n*Vt*(R_series+R_shunt)))*np.exp((R_shunt*(V+Jph*R_series+J0*R_series))/(n*Vt*(R_series+R_shunt)))) # check equation (2) in the paper
    w = w.real # remove the imaginary part
    Current = -(V/(R_series+R_shunt)) - (n*Vt/R_series) * w + ((R_shunt*(J0+Jph))/(R_series+R_shunt))

    return -Current.real