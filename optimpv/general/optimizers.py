
import numpy as np




######################################################################
#################### BOAR Optimizer Class ############################
######################################################################
# Authors: 
# Larry Lueer (https://github.com/larryluer)
# Vincent M. Le Corre (https://github.com/VMLC-PV)
# (c) i-MEET, University of Erlangen-Nuremberg, 2021-2022-2023 


# Import libraries
import os,copy,warnings
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy
from sklearn.metrics import mean_squared_error,mean_squared_log_error,mean_absolute_error,mean_absolute_percentage_error
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import curve_fit,basinhopping,dual_annealing
from functools import partial
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
# Import boar libraries
from boar.core.funcs import sci_notation
from boar.core.funcs import get_unique_X


class BoarOptimizer():
    """ Provides a default class for the different optimizers in BOAR. \\
    This class is not intended to be used directly, but rather to be inherited by the different optimizer classes. \\
    It provides the basic functionality for the optimizer classes, such as the different objective functions, \\
    the functions to handle the Fitparam() objects, and the functions to handle the different plotting options. \\

    
    """    
    # a class for multi-objective optimization
    def __init__(self) -> None:
        pass

    
    def params_w(x,params):
        """Populate the Fitparam() objects with the values from the list x

        Parameters
        ----------
        x : list of float or int or str or bool
            list of values to populate the Fitparam() objects
        params : list of Fitparam() objects
            list of Fitparam() objects

        Raises
        ------
        ValueError
            If the value_type of the parameter is not 'float', 'int', 'str', 'cat', 'sub' or 'bool'
        """    

        count = 0
        for param in params:
            if param.type != 'fixed':
                if param.value_type == 'float':
                    param.value = x[count] * param.fscale
                elif param.value_type == 'int':
                    param.value = int(x[count]) * param.stepsize
                elif param.value_type == 'cat' or param.value_type == 'sub' or param.value_type == 'str': 
                    param.value = x[count]
                elif param.value_type == 'bool':
                    param.value = bool(x[count])
                else:
                    raise ValueError('Failed to convert parameter name: {} to Ax format'.format(param.name))

                count += 1

        return params
 


    def format_func(self,value, tick_number):
        """Format function for the x and y axis ticks  
        to be passed to axo[ii,jj].xaxis.set_major_formatter(plt.FuncFormatter(format_func))  
        to get the logarithmic ticks looking good on the plot  

        Parameters
        ----------
        value : float
            value to convert
        tick_number : int
            tick position

        Returns
        -------
        str
            string representation of the value in scientific notation
        """        
        return sci_notation(10**value, sig_fig=-1)


    