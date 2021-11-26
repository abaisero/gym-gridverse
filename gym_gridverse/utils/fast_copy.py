import pickle
from typing import TypeVar

T = TypeVar('T')
"""generic type"""


def fast_copy(x: T) -> T:
    """returns a deep copy of a generic python object, faster than deepcopy"""
    return pickle.loads(pickle.dumps(x))
