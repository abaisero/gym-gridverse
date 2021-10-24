from typing import Callable, Optional, Type

# library-level debugging flag
_gv_debug: Optional[bool] = None


def reset_gv_debug(debug: Optional[bool] = None) -> bool:
    """Sets the library-wide debugging boolean."""
    global _gv_debug
    _gv_debug = debug if debug is not None else __debug__
    return _gv_debug


def gv_debug() -> bool:
    """Gets the library-wide debugging boolean.

    Used to bypass expensive type and value checks at runtime.  By default (if
    :py:func:`~gym_gridverse.debugging.reset_gv_debug` was not called), the
    value of `__debug__` is used.
    """
    return reset_gv_debug() if _gv_debug is None else _gv_debug


def checkraise(
    condition_f: Callable[[], bool],
    error_type: Type[Exception],
    error_message_fmt: str,
    *args,
    **kwargs
):
    if gv_debug() and not condition_f():
        raise error_type(error_message_fmt.format(*args, **kwargs))
