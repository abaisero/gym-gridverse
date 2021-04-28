from typing import Callable, Optional, Type

# library-level debugging flag
_gv_debug: Optional[bool] = None


def reset_gv_debug(debug: Optional[bool] = None) -> bool:
    global _gv_debug  # pylint: disable=global-statement
    _gv_debug = debug if debug is not None else __debug__
    return _gv_debug


def get_gv_debug() -> bool:
    return reset_gv_debug() if _gv_debug is None else _gv_debug


def checkraise(
    condition_f: Callable[[], bool],
    error_type: Type[Exception],
    error_message_fmt: str,
    *args,
    **kwargs
):
    if get_gv_debug() and not condition_f():
        raise error_type(error_message_fmt.format(*args, **kwargs))
