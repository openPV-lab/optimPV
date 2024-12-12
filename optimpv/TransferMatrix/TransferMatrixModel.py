"""Transfer Matrix Model"""
# Note: This class is inspired by the https://github.com/erichoke/Stanford repository and the paper:
# Burkhard, G.F., Hoke, E.T. and McGehee, M.D. (2010), Accounting for Interference, Scattering, and Electrode Absorption to Make Accurate Internal Quantum Efficiency Measurements in Organic and Other Thin Solar Cells. Adv. Mater., 22: 3293-3297. https://doi.org/10.1002/adma.201000883
######### Package Imports #########################################################################

import os, uuid, sys, copy
import numpy as np
import pandas as pd
from math import ceil
from copy import deepcopy
from scipy import constants
from scipy.interpolate import interp1d

# physical constants
q = constants.value(u'elementary charge')
c = constants.value(u'speed of light in vacuum')
h = constants.value(u'Planck constant')

######### Function Definitions ####################################################################
def openFile(fname):
    """ opens files and returns a list split at each new line

    Parameters
    ----------
    fname : string
        path to the file

    Returns
    -------
    fd : list
        list of lines in the file

    Raises
    ------
    ValueError
        Target is not a readable file

    """    
    fd = []
    if os.path.isfile(fname):
        fn = open(fname, 'r')
        fdtmp = fn.read()
        fdtmp = fdtmp.split('\n')
        # clean up line endings
        for f in fdtmp:
            f = f.strip('\n')
            f = f.strip('\r')
            fd.append(f)
        # make so doesn't return empty line at the end
        if len(fd[-1]) == 0:
            fd.pop(-1)
    else:
        print("%s Target is not a readable file" % fname)
    return fd

def get_ntotal(matName,lambdas,mat_dir):
    """ get the complex refractive index of a material from a file

    Parameters
    ----------
    matName : string
        name of the material in the matdata folder
    lambdas : list
        list of wavelengths in nm

    Returns
    -------
    ntotal : list
        list of complex refractive index values
    """      
    matPrefix		= 'nk_'		    # materials data prefix  
    fname = os.path.join(mat_dir,'%s%s.txt' % (matPrefix,matName))
    matHeader = 0
    # check number of lines with strings in the header
    for line in openFile(fname):
        # check if line starts with a number
        if line[0].isdigit():
            break
        else:
            matHeader += 1
    
    fdata = openFile(fname)[matHeader:]
    fdata = pd.read_csv(fname, sep=r'\s+', header='infer')
    lambList	= np.asarray(fdata['lambda'])
    nList		= np.asarray(fdata['n'])
    kList		= np.asarray(fdata['k'])
    # # get data from the file
    # lambList	= []
    # nList		= []
    # kList		= []

    # for idx,l in enumerate(fdata):
    #     # remove any whitespace at the beginning of the line end in the end
    #     l = l.strip()
    #     wl , n , k = l.split(' ')
    #     wl , n , k = float(wl) , float(n) , float(k)
    #     lambList.append(wl)
    #     nList.append(n)
    #     kList.append(k)
    # make interpolation functions
    int_n	= interp1d(lambList,nList,fill_value='extrapolate')
    int_k	= interp1d(lambList,kList,fill_value='extrapolate')
    # interpolate data
    # check if the wavelengths are within the range of the data with precision of float
    epsilon = np.finfo(float).eps
    if np.min(lambdas) < np.min(lambList) - epsilon or np.max(lambdas) > np.max(lambList) + epsilon:
        print(np.min(lambdas),np.min(lambList),np.max(lambdas),np.max(lambList))
        raise ValueError('Wavelengths out of range for the %s layer. Please either change the wavelength range or provide and nk file with the proper range.' % matName)
    kintList	= int_k(lambdas)
    nintList	= int_n(lambdas)
    # make ntotal
    ntotal = []
    for i,n in enumerate(nintList):
        nt = complex(n,kintList[i])
        ntotal.append(nt)
    return ntotal

def I_mat(n1,n2):
    """ calculate the interface matrix

    Parameters
    ----------
    n1 : float
        refractive index of the first material
    n2 : float
        refractive index of the second material

    Returns
    -------
    ret : array
        interface matrix
    """        
    r = (n1-n2)/(n1+n2)
    t = (2*n1)/(n1+n2)
    ret = np.array([[1,r],[r,1]],dtype=complex)
    ret = ret / t
    return ret

def L_mat(n,d,l):
    """ calculate the propagation matrix

    Parameters
    ----------
    n : array
        complex refractive index of the material
    d : float
        thickness of the material
    l : float
        wavelength

    Returns
    -------
    L : array
        propagation matrix
    """        

    xi = (2*np.pi*d*n)/l
    L = np.array( [ [ np.exp(complex(0,-1.0*xi)),0] , [0,np.exp(complex(0,xi))] ] )
    return L


def TMM(parameters, layers, thicknesses, lambda_min, lambda_max, lambda_step, x_step, activeLayer, spectrum, mat_dir,photopic_file=None): 
    """ Calculate the Jsc, AVT or LUE for a multilayer stack

    Parameters
    ----------
    parameters : dict
        dictionary of parameters note that all parameters must be in the form of 'd_i' or 'nk_i' where i is the index of the layer and the everything must be in SI units.
    layers : list
        list of material names in the stack. Note that this names will be used to find the refractive index files in the mat_dir. The filenames must be in the form of 'nk_materialname.txt'
    thicknesses : list
        list of thicknesses of the layers in the stack in meters
    lambda_min : float
        start wavelength in m
    lambda_max : float
        stop wavelength in m
    lambda_step : float
        wavelength step in m
    x_step : float
        step size for the x position in the stack in m
    activeLayer : int
        index of the active layer in the stack, i.e. the layer where the generation profile will be calculated. Counting starts at 0.
    spectrum : string
        name of file that contains the spectrum.
    mat_dir : string
        path to the directory where the refractive index files and the spectrum file are located.
    photopic_file : string, optional
        name of the file that contains the photopic response (must be in the same directory as the refractive index files), by default None

    Returns
    -------
    Jsc : float
        Short circuit current
    AVT : float
        Average visible transmittance
    LUE : float
        Light utilization efficiency

    Raises
    ------
    ValueError
        Wrong indices for the thicknesses
    ValueError
        Wrong indices for the complex refractive index
    ValueError
        Wavelengths out of range for the layer
    ValueError
        photopic_file must be defined to calculate AVT or LUE

    """     

    if photopic_file is not None:
        calculate_AVT_LUE = True
    else:
        calculate_AVT_LUE = False

    # prepare the stack
    pnames = list(parameters.keys())

    # Read the parameters
    dnames = [p for p in pnames if p.startswith('d_')] #find parameters that start with 'd_'
    dindices = [int(p.split('_')[1]) for p in dnames] # read index after 'd_' for these parameters
    nknames = [p for p in pnames if p.startswith('nk_')]
    nkindices = [int(p.split('_')[1]) for p in nknames]

    # check that all indexes in dinices are in nkindices are below the number of layers
    maxindex = len(layers)
    if any([i>maxindex for i in dindices]):
        raise ValueError('dindices must be below the number of layers')
    if any([i>maxindex for i in nkindices]):
        raise ValueError('nkindices must be below the number of layers')

    t = deepcopy(thicknesses)
    lambdas	= np.arange(lambda_min,lambda_max+lambda_step,lambda_step,np.float64)
    layers = deepcopy(layers)
    # x_step = deepcopy(self.x_step)
    # activeLayer = deepcopy(self.activeLayer)

    # update the thicknesses
    for i in dindices:
        # thicknesses[i] = [p.val for p in params if p.name == 'd_'+str(i)][0]
        if 'd_'+str(i) in parameters.keys():
            thicknesses[i] = parameters['d_'+str(i)]
        if 'd_'+str(i) in parameters.keys():
            t[i] = parameters['d_'+str(i)]
        # t[i] = [p.val for p in params if p.name == 'd_'+str(i)][0]
    # update the nk values
    for i in nkindices:
        # layers[i] = [p.val for p in params if p.name == 'nk_'+str(i)][0]
        # layers[i] = [p.val for p in params if p.name == 'nk_'+str(i)][0]
        if 'nk_'+str(i) in parameters.keys():
            layers[i] = parameters['nk_'+str(i)]

    
    # load and interpolate AM1.5G Data
    am15 = pd.read_csv(spectrum, sep=r'\s+', header='infer')
    am15_xData = np.asarray(am15['lambda'])
    am15_yData = np.asarray(am15['I'])
    am15_interp = interp1d(am15_xData,am15_yData,'linear')
    am15_int_y = am15_interp(lambdas)

    # load and interpolate human eye response
    if photopic_file is not None:
        photopic_file = os.path.join(photopic_file)
        photopic_data = pd.read_csv(photopic_file, sep=r'\s+', header='infer')
        photopic_xData = np.asarray(photopic_data['lambda'])
        photopic_yData = np.asarray(photopic_data['photopic'])
        photopic_interp = interp1d(photopic_xData,photopic_yData,'linear')
        photopic_int_y  = photopic_interp(lambdas)

    # ------ start actual calculation  --------------------------------------
    

    # initialize an array
    n = np.zeros((len(layers),len(lambdas)),dtype=complex)

    # load index of refraction for each material in the stack
    for i,l in enumerate(layers):
        ni = np.array(get_ntotal(l,lambdas,mat_dir))
        n[i,:] = ni

    # calculate incoherent power transmission through substrate

    T_glass = abs((4.0*1.0*n[0,:])/((1+n[0,:])**2))
    R_glass = abs((1-n[0,:])/(1+n[0,:]))**2

    # calculate transfer matrices, and field at each wavelength and position
    t[0] 		= 0
    t_cumsum	= np.cumsum(t)
    x_pos		= np.arange((x_step/2.0),sum(t),x_step)
    # get x_mat
    comp1	= np.kron(np.ones( (len(t),1) ),x_pos)
    comp2	= np.transpose(np.kron(np.ones( (len(x_pos),1) ),t_cumsum))
    x_mat 	= sum(comp1>comp2,0) 	# might need to get changed to better match python indices

    R		= lambdas*0.0
    T2		= lambdas*0.0
    E		= np.zeros( (len(x_pos),len(lambdas)),dtype=complex )

    # start looping
    for ind,l in enumerate(lambdas):
        # calculate the transfer matrices for incoherent reflection/transmission at the first interface
        S = I_mat(n[0,ind],n[1,ind])
        for matind in np.arange(1,len(t)-1):
            mL = L_mat( n[matind,ind] , t[matind] , lambdas[ind] )
            mI = I_mat( n[matind,ind] , n[matind+1,ind])
            S  = np.asarray(np.asmatrix(S)*np.asmatrix(mL)*np.asmatrix(mI))
        R[ind] = abs(S[1,0]/S[0,0])**2
        T2[ind] = abs((2/(1+n[0,ind])))/np.sqrt(1-R_glass[ind]*R[ind])

        # this is not the transmittance! 
        # good up to here
        # calculate all other transfer matrices
        for material in np.arange(1,len(t)):
                xi = 2*np.pi*n[material,ind]/lambdas[ind]
                dj = t[material]
                x_indices	= np.nonzero(x_mat == material)
                x			= x_pos[x_indices]-t_cumsum[material-1]
                # Calculate S_Prime
                S_prime		= I_mat(n[0,ind],n[1,ind])
                for matind in np.arange(2,material+1):
                    mL = L_mat( n[matind-1,ind],t[matind-1],lambdas[ind] )
                    mI = I_mat( n[matind-1,ind],n[matind,ind] )
                    S_prime  = np.asarray( np.asmatrix(S_prime)*np.asmatrix(mL)*np.asmatrix(mI) )
                # Calculate S_dprime (double prime)
                S_dprime	= np.eye(2)
                for matind in np.arange(material,len(t)-1):
                    mI	= I_mat(n[matind,ind],n[matind+1,ind])
                    mL	= L_mat(n[matind+1,ind],t[matind+1],lambdas[ind])
                    S_dprime = np.asarray( np.asmatrix(S_dprime) * np.asmatrix(mI) * np.asmatrix(mL) )
                # Normalized Electric Field Profile
                num = T2[ind] * (S_dprime[0,0] * np.exp( complex(0,-1.0)*xi*(dj-x) ) + S_dprime[1,0]*np.exp(complex(0,1)*xi*(dj-x)))
                den = S_prime[0,0]*S_dprime[0,0]*np.exp(complex(0,-1.0)*xi*dj) + S_prime[0,1]*S_dprime[1,0]*np.exp(complex(0,1)*xi*dj)
                
                E[x_indices,ind] = num / den
    # overall Reflection from device with incoherent reflections at first interface
    Reflectance = R_glass+T_glass**2*R/(1-R_glass*R)
    
    # Absorption coefficient in 1/cm
    a = np.zeros( (len(t),len(lambdas)) )
    for matind in np.arange(1,len(t)):
        a[matind,:] = ( 4 * np.pi * np.imag(n[matind,:]) ) / ( lambdas ) #* 1.0e-7 )

    # Absorption
    Absorption = np.zeros( (len(t),len(lambdas)) )
    for matind in np.arange(1,len(t)):
        Pos 		= np.nonzero(x_mat == matind)
        AbsRate 	= np.tile( (a[matind,:] * np.real(n[matind,:])),(len(Pos),1)) * (abs(E[Pos,:])**2)
        Absorption[matind,:] = np.sum(AbsRate,1)*x_step#*1.0e-7

    # Transmittance
    Transmittance = 1 - Reflectance - np.sum(Absorption,0)
    Transmittance[Transmittance<0] = 0 # set negative values to zero

    # calculate generation profile
    ActivePos = np.nonzero(x_mat == activeLayer)
    tmp1	= (a[activeLayer,:]*np.real(n[activeLayer,:])*am15_int_y)
    Q	 	= np.tile(tmp1,(np.size(ActivePos),1))*(abs(E[ActivePos,:])**2)

    # Exciton generation rate
    Gxl		= (Q)*np.tile( (lambdas) , (np.size(ActivePos),1))/(h*c)

    if len(lambdas) == 1:
        lambda_step = 1
    else:
        lambda_step = (sorted(lambdas)[-1] - sorted(lambdas)[0])/(len(lambdas) - 1)
    Gx		= np.sum(Gxl,2)*lambda_step
    
    # calculate Jsc 
    Jsc = np.sum(Gx)*x_step*q

    # calculate AVT and LUE
    if calculate_AVT_LUE:
        AVT = sum(am15_int_y * photopic_int_y * Transmittance)/sum(am15_int_y * photopic_int_y)
        LUE = Jsc * AVT
    else:
        AVT = None
        LUE = None

    return Jsc,AVT,LUE

