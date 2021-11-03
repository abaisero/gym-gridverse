import pickle
from typing import TypeVar

T = TypeVar('T')


def fast_copy(x: T) -> T:
    return pickle.loads(pickle.dumps(x))
