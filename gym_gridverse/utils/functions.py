import importlib
from typing import Any, Collection, Container, Dict

from gym_gridverse.debugging import checkraise


def checkraise_kwargs(kwargs: Dict[str, Any], required_keys: Collection[str]):
    for key in required_keys:
        checkraise(
            lambda: key in kwargs,
            ValueError,
            'missing keyword argument `{}`',
            key,
        )


def select_kwargs(kwargs: Dict[str, Any], keys: Container[str]):
    return {key: value for key, value in kwargs.items() if key in keys}


def is_custom_function(name):
    return ':' in name


def get_custom_function(name):
    module_name, function_name = name.split(':')
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function
