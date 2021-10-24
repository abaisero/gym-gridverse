import importlib
from typing import Any, Collection, Container, Dict


def checkraise_kwargs(kwargs: Dict[str, Any], required_keys: Collection[str]):
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f'missing keyword argument `{key}`')


def select_kwargs(kwargs: Dict[str, Any], keys: Container[str]):
    return {key: value for key, value in kwargs.items() if key in keys}


def is_custom_function(name: str) -> bool:
    return ':' in name


def import_custom_function(name: str) -> str:
    module_name, function_name = name.split(':')
    importlib.import_module(module_name)
    return function_name


def get_custom_function(name: str):
    module_name, function_name = name.split(':')
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function
