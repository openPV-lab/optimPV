"""FitParams class"""
# Note: This class is inspired by the https://github.com/i-MEET/boar/ package
######### Package Imports #########################################################################

import numpy as np

######### Function Definitions ####################################################################

class FitParam():
    def __init__(self, name = '', type = 'range', value_type = 'float', value = None, bounds = None, values = None,  start_value = None, log_scale = False, rescale = False, fscale = None, stepsize = None, display_name='', unit='', axis_type = None, std = None,encoding = None, force_log = False):
        """ Fitparam class object

        Parameters
        ----------
        name : str, optional
            name by which object can be retrived, by default ''
        type : str, optional
            type of parameter ('range', 'choice' or 'fixed'), by default 'range'
        value_type : str, optional
            type of value ('float', 'int', 'cat', 'sub', 'str', 'bool'), by default 'float'
        value : float or int or str, optional
            value of the parameter (used when type='fixed' or contained the optimized value), by default None
        bounds : list, optional
            lower and upper bounds of the parameter (type='range'), by default None
        values : list, optional
            possible values of the parameter (type='choice'), by default None
        startVal : float, optional
            starting guess for the optimization, by default None
        log_scale : bool, optional
            if True the parameter is optimized in log scale, by default False
        rescale : bool, optional
            if True the parameter is rescaled by fscale, by default True
        fscale : float, optional
            order of magnitude of the parameter/scaling factor, by default None
        stepsize : float, optional
            stepsize for integer parameters (value_type = 'int') can be used by the model to transform the integer given by the optimizer to the actual value, i.e. the optimizer sees integers but the model uses value*stepsize, by default None
        display_name : str, optional
            name to be displayed in plots, by default ''
        unit : str, optional
            unit of the parameter, by default ''
        axis_type : str, optional
            Set the type of scale and formatting for the axis of the plots ('linear' or 'log') if left to None we use log_scale, by default None
        std : list of float, optional
            standard deviation if returned by optimization, by default None
        encoding : str, optional
            encoding of the parameter (not yet in use), by default None
        force_log : bool, optional
            take the log of the parameter prior to passing it to the model, ignoring the log_scale and rescale, by default False

        Raises
        ------
        ValueError
            if axis_type is not 'linear', 'lin', 'logarithmic' or 'log'
            if value_type is not 'float', 'int', 'str', 'cat', 'sub' or 'bool'
            if log_scale is not a bool
            if fscale is not None, int or float
            if type is not 'range', 'choice' or 'fixed'
            if bounds is not defined for range type
            if bounds is not a list of length 2
            if value is not defined for fixed type

        Examples
        --------
        >>> from optimpv.general import FitParam
        >>> param = FitParam(name='param1', type='range', value_type='float', bounds=[1e-2, 1], log_scale=True)

        """        

        self.name = name
        self.type = type
        self.value_type = value_type
        self.value = value
        self.bounds = bounds
        self.values = values
        self.start_value = value if start_value is None else start_value
        self.log_scale = log_scale
        self.fscale = fscale
        self.stepsize = stepsize
        self.rescale = rescale
        self.display_name = display_name if display_name else name
        self.unit = unit
        self.full_name = f"{self.display_name} [{self.unit}]" if unit else self.display_name
        if axis_type is not None:
            self.axis_type = axis_type
        else:
            self.axis_type = 'log' if log_scale else 'linear'
        self.std = std
        self.encoding = encoding
        self.force_log = force_log

        # Checks
        if self.axis_type not in ['linear', 'log', 'lin', 'logarithmic']:
            raise ValueError("axis_type must be 'linear', 'lin', 'logarithmic' or 'log'")
        if self.value_type not in ['float', 'int', 'str', 'cat', 'sub', 'bool']:
            raise ValueError("value_type must be 'float', 'int', 'str', 'cat', 'sub' or 'bool'")
        # check is log_scale is a bool
        if not isinstance(self.log_scale, bool):
            raise ValueError("log_scale must be a bool")
        # correct axis_type to match matplotlib
        if self.axis_type == 'lin':  
            self.axis_type = 'linear'
        elif self.axis_type == 'logarithmic':
            self.axis_type = 'log'

        # Check that fscale is either None, int or float
        if self.fscale is not None and not isinstance(self.fscale, (int, float)):
            raise ValueError('fscale must be None, int or float')
        
        if self.type == 'range':
            if self.bounds is None:
                raise ValueError('bounds must be defined for range type')
            if len(self.bounds) != 2:
                raise ValueError('bounds must be a list of length 2')
            if self.value_type == 'int':
                self.bounds = [int(self.bounds[0]), int(self.bounds[1])]
        elif self.type == 'choice':
            if self.values is None:
                raise ValueError('values must be defined for choice type')
        elif self.type == 'fixed':
            if self.value is None:
                raise ValueError('value must be defined for fixed type')
        else:
            raise ValueError('type must be range, choice or fixed')      

        if self.value_type == 'float' or self.value_type == 'int':
            if self.value > self.bounds[1] or self.value < self.bounds[0]:
                raise ValueError('value must be within bounds')
            if self.start_value is not None:
                if self.start_value > self.bounds[1] or self.start_value < self.bounds[0]:
                    raise ValueError('start_value must be within bounds')

        if self.value_type == 'float':
            
            if self.force_log:
                # self.bounds = [np.log10(self.bounds[0]), np.log10(self.bounds[1])]
                self.log_scale = False
                self.rescale = False
                self.fscale = 1
            else:
                if not self.rescale:
                    self.fscale = 1
                if self.fscale is None:
                    
                    if self.start_value is not None:
                        self.fscale = 10**np.floor(np.log10(abs(self.start_value)))
                    else:
                        if self.type == 'range':
                            if self.bounds[0] != 0 and self.bounds[1] != 0:
                                self.fscale = 10**np.floor(np.log10(np.sqrt(self.bounds[0]*self.bounds[1])))# geometric mean of the bounds
                            else:
                                self.fscale = 10**np.floor(np.log10(abs(self.bounds[0] + self.bounds[1]))) # take the one that is not zero
                        elif self.type == 'choice':
                            self.fscale = 10**np.floor(np.mean(np.log10([abs(val) for val in self.values]))) # geometric mean of the values
                        
        else:
            self.fscale = 1 # not necessary for int, cat, sub, str, bool but we set it to 1 to avoid errors

        if self.value_type == 'int' and self.stepsize is None:
            self.stepsize = 1
            
        
    def __repr__(self):
        """Representation of the Fitparam object with all attributes
        Returns
        -------
        str
            representation of the Fitparam object with all attributes
        """
        return str(vars(self))