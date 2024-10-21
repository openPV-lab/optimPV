"""FitParams class"""
# Note: This class is inspired by the https://github.com/i-MEET/boar/ package
######### Package Imports #########################################################################

import numpy as np

######### Function Definitions ####################################################################

class Fitparam():
    def __init__(self, name = '', value = None, bounds = None, values = None, Fixed = False, start_value = None, log_scale = False, value_type = 'float', p0m = None, rescale = True, stepsize = None, display_name='', unit='', axis_type = None, std = 0,encoding = None):
        """ Fitparam class object

        Parameters
        ----------
        name : str, optional
            name by which object can be retrived, by default ''
        val : float, optional
            achieved value after the optimizaton, by default 1.0
        lims : list, optional
            hard limits, by default []
        Fixed : bool, optional
            if True, parameter is considered fixed, by default False
        startVal : float, optional
            starting guess for the optimization, by default None
        p0m : float, optional
            order of magnitude of the parameter/scaling factor, by default None
        std : float, optional
            standard deviation if returned by optimization, by default 0
        display_name : str, optional
            name to be displayed in plots, by default ''
        unit : str, optional
            unit of the parameter, by default ''
        log_scale : bool, optional
            Interpretation by optimizer ('linear' or 'log'), by default False
        axis_type : str, optional
            Set the type of scale and formatting for the axis of the plots ('linear' or 'log') if left to None we use log_scale, by default None
        value_type : str, optional
            type of the parameter, can be 'float' or 'int' or 'str', by default 'float'
        rescale : bool, optional
            if log_scale = 'linear' it rescales the parameter to the order of magnitude of the start_value if p0m is not defined, for now this does not affect the results if log_scale = 'log', by default True
        stepsize : float, optional
            stepsize for integer parameters (value_type = 'int'), by default None

        Raises
        ------
        ValueError
            range_type must be 'linear', 'lin', 'logarithmic' or 'log'
        ValueError
            log_scale must be 'linear', 'lin', 'logarithmic' or 'log'
        ValueError
            value_type must be 'float', 'int', 'cat' or 'sub'
        ValueError
            p0m must be None, int or float
        """        

        self.name = name
        self.display_name = display_name if display_name else name
        self.unit = unit
        self.full_name = f"{self.display_name} [{self.unit}]" if unit else self.display_name
        self.log_scale = log_scale
        self.axis_type = 'log' if log_scale else 'linear'
        self.value_type = value_type
        self.stepsize = stepsize

        # Check if limits are valid
        if self.axis_type not in ['linear', 'log', 'lin', 'logarithmic']:
            raise ValueError("axis_type must be 'linear', 'lin', 'logarithmic' or 'log'")
        if self.value_type not in ['float', 'int', 'cat', 'sub']:
            raise ValueError("value_type must be 'float', 'int' or 'cat'")
        
        if self.axis_type == 'lin':  # correct for improper axis_type
            self.axis_type = 'linear'
        elif self.axis_type == 'logarithmic':
            self.axis_type = 'log'

        if self.value_type != 'cat':
            self.start_value = value if start_value is None else start_value
            self.value = value
        else:
            self.start_value = value if start_value is None else start_value
            self.value = value

        self.p0m = p0m

        # Check that p0m is either None, int or float
        if self.p0m is not None and not isinstance(self.p0m, (int, float)):
            raise ValueError('p0m must be None, int or float')

        self.bounds = bounds

        if self.value_type == 'int':
            self.bounds = (int(self.bounds[0]), int(self.bounds[1]))

        self.std = std
        self.Fixed = Fixed
        self.rescale = rescale

    def __str__(self):
        """ String representation of the Fitparam object with all attributes when printed
        Returns
        -------
        str
            string representation of the Fitparam object with all attributes
        """
        all_attributes = vars(self)
        attribute_string = "\n".join(f"{key}: {value}" for key, value in all_attributes.items() if not key.startswith("__"))
        return attribute_string
    
    def __repr__(self):
        """ String representation of the Fitparam object

        Returns
        -------
        str
            string representation of the Fitparam object
        """   
        return f"{self.__class__.__name__}({self.name}, {self.value}), bounds = {self.bounds}, Fixed = {self.Fixed}, start_value = {self.start_value}, log_scale = {self.optim_type}, value_type = {self.value_type}, p0m = {self.p0m}, rescale = {self.rescale}, stepsize = {self.stepsize}, display_name = {self.display_name}, unit = {self.unit}, axis_type = {self.axis_type}, std = {self.std}"
    
    def __dict__(self):
        """ Dictionary representation of the Fitparam object

        Returns
        -------
        dict
            dictionary representation of the Fitparam object
        """   
        return vars(self)
