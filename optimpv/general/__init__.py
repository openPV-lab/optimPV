# import os
# import importlib.util
# import pkgutil

# # Path to the directory of this file
# package_dir = os.path.dirname(__file__)

# # Loop through all modules in the current directory
# for (_, module_name, _) in pkgutil.iter_modules([package_dir]):
#     # Construct the full path to the module
#     module_path = os.path.join(package_dir, module_name + '.py')
#     # Import the module
#     spec = importlib.util.spec_from_file_location(module_name, module_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     # Add all functions from the module to globals() so they can be imported directly
#     for attribute_name in dir(module):
#         attribute = getattr(module, attribute_name)
#         if callable(attribute):
#             globals()[attribute_name] = attribute

# # from pySIMsalabim.general import *
# # from pySIMsalabim.utils import *
# # from pySIMsalabim.device_parameters import *