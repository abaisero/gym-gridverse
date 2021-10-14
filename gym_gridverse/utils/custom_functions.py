import importlib


def is_custom_function(name):
    return ':' in name


def get_custom_function(name):
    module_name, function_name = name.split(':')
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function
