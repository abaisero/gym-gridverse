import importlib


def is_custom(name: str) -> bool:
    """Checks if input name indicates a custom function/object

    A custom name is expressed as `<module_name>:<stripped_name>`.

    Args:
        name (str):  Potentially custom name
    Returns:
        bool:  True iff the input name is custom
    """
    return ':' in name


def import_custom(name: str) -> str:
    """Imports custom module and returns stripped name.

    A custom name is expressed as `<module_name>:<stripped_name>`.

    Args:
        name (str):  Custom name
    Returns:
        str:  The stripped name
    """
    module_name, stripped_name = name.split(':')
    importlib.import_module(module_name)
    return stripped_name


def import_if_custom(name: str) -> str:
    """Conditionally imports custom module and returns stripped name.

    Combines :py:func:`~gym_gridverse.utils.custom.is_custom` and
    :py:func:`~gym_gridverse.utils.custom.import_custom`.

    Args:
        name (str):  Potentially custom name
    Returns:
        str:  If custom, the stripped name, else the input name
    """
    return import_custom(name) if is_custom(name) else name
