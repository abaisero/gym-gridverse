from typing import Any, Collection, Container, Dict


def checkraise_kwargs(kwargs: Dict[str, Any], required_keys: Collection[str]):
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f'missing keyword argument `{key}`')


def select_kwargs(kwargs: Dict[str, Any], keys: Container[str]):
    return {key: value for key, value in kwargs.items() if key in keys}
