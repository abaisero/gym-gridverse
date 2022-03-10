from typing import List, Optional, Sequence, TypeVar

import numpy.random as rnd

# library-level generator, used if one is not provided (e.g. by environment)
_gv_rng: Optional[rnd.Generator] = None


def make_rng(seed: Optional[int] = None) -> rnd.Generator:
    """make a new rng object"""
    return rnd.default_rng(seed)


def reset_gv_rng(seed: Optional[int] = None) -> rnd.Generator:
    """reset the gym-gridverse module rng"""
    global _gv_rng
    _gv_rng = make_rng(seed)
    return _gv_rng


def get_gv_rng() -> rnd.Generator:
    """get (and reset if necessary) gym-gridverse module rng"""
    return reset_gv_rng() if _gv_rng is None else _gv_rng


def get_gv_rng_if_none(rng: Optional[rnd.Generator]) -> rnd.Generator:
    """get gym-gridverse module rng if input is None"""
    return get_gv_rng() if rng is None else rng


T = TypeVar('T')


def choice(rng: rnd.Generator, data: Sequence[T]) -> T:
    """utility function to help typing, only valid for size 1"""
    i = rng.choice(len(data))
    return data[i]


def choices(
    rng: rnd.Generator, data: Sequence[T], *, size: int, **kwargs
) -> List[T]:
    """utility function to help typing"""
    indices = rng.choice(len(data), size=size, **kwargs)
    return [data[i] for i in indices]


def shuffle(rng: rnd.Generator, data: Sequence[T]) -> List[T]:
    indices = list(range(len(data)))
    # NOTE: faster than rng.choice
    rng.shuffle(indices)
    return [data[i] for i in indices]
