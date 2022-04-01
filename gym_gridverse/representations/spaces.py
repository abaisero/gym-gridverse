from __future__ import annotations

import enum
from typing import Tuple, cast

import numpy as np


class SpaceType(enum.Enum):
    CATEGORICAL = 0
    DISCRETE = enum.auto()
    CONTINUOUS = enum.auto()


def is_dtype_integer(x: np.ndarray) -> bool:
    """checks if array has an integer type

    Args:
        x (numpy.ndarray): x

    Returns:
        bool:
    """
    return np.issubdtype(x.dtype, np.integer)


def is_dtype_floating(x: np.ndarray) -> bool:
    """checks if array has a floating type

    Args:
        x (numpy.ndarray): x

    Returns:
        bool:
    """
    return np.issubdtype(x.dtype, np.floating)


def is_dtype_compatible(x: np.ndarray, space_type: SpaceType) -> bool:
    """checks if array type is compatible with space type

    Args:
        x (numpy.ndarray): x
        space_type (SpaceType): space_type

    Returns:
        bool:
    """
    # TODO: test
    if space_type is SpaceType.CATEGORICAL:
        return is_dtype_integer(x)

    if space_type is SpaceType.DISCRETE:
        return is_dtype_integer(x)

    if space_type is SpaceType.CONTINUOUS:
        return is_dtype_floating(x)

    raise ValueError(f'invalid SpaceType {space_type}')


class Space:
    def __init__(
        self,
        space_type: SpaceType,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ):
        """initializes a bounded space

        `lower_bound` and `upper_bound` must have the same shape, and a dtype
        compatible with the space_type.  Each element of `lower_bound` must be
        lower or equal to the corresponding element of `upper_bound`.

        Args:
            space_type (SpaceType): space_type
            lower_bound (numpy.ndarray): lower_bound
            upper_bound (numpy.ndarray): upper_bound
        """
        if not is_dtype_compatible(lower_bound, space_type):
            raise ValueError(
                f'incompatible lower bound dtype {lower_bound.dtype}'
            )
        if not is_dtype_compatible(upper_bound, space_type):
            raise ValueError(
                f'incompatible upper bound dtype {upper_bound.dtype}'
            )
        if lower_bound.shape != upper_bound.shape:
            raise ValueError(
                f'incompatible bound shapes {lower_bound.shape} {upper_bound.shape}'
            )
        if np.any(lower_bound > upper_bound):
            raise ValueError('incompatible bound values')

        self.space_type = space_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.lower_bound.shape

    def contains(self, x: np.ndarray) -> bool:
        """checks if array is of the correct shape and type, and within the space bounds

        Args:
            x (numpy.ndarray): x

        Returns:
            bool:
        """
        return (
            x.shape == self.shape
            and is_dtype_compatible(x, self.space_type)
            and cast(bool, np.all(self.lower_bound <= x))
            and cast(bool, np.all(x <= self.upper_bound))
        )

    def __eq__(self, other):
        if isinstance(other, Space):
            return (
                self.space_type == other.space_type
                and np.array_equal(self.lower_bound, other.lower_bound)
                and np.array_equal(self.upper_bound, other.upper_bound)
            )
        return NotImplemented

    @staticmethod
    def make_categorical_space(upper_bound: np.ndarray) -> Space:
        """initializes a bounded categorical space

        `upper_bound` must have an integer dtype.  Each element of
        `upper_bound` must be non-negative.

        Args:
            upper_bound (numpy.ndarray): upper_bound
        """
        lower_bound = np.zeros_like(upper_bound)
        return Space(
            SpaceType.CATEGORICAL,
            lower_bound,
            upper_bound,
        )

    @staticmethod
    def make_discrete_space(
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> Space:
        """initializes a bounded discrete space

        `lower_bound` and `upper_bound` must have the same shape, and an
        integer dtype.  Each element of `lower_bound` must be lower or equal to
        the corresponding element of `upper_bound`.

        Args:
            space_type (SpaceType): space_type
            lower_bound (numpy.ndarray): lower_bound
            upper_bound (numpy.ndarray): upper_bound
        """
        return Space(
            SpaceType.DISCRETE,
            lower_bound,
            upper_bound,
        )

    @staticmethod
    def make_continuous_space(
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> Space:
        """initializes a bounded continuous space

        `lower_bound` and `upper_bound` must have the same shape, and a
        floating dtype.  Each element of `lower_bound` must be lower or equal
        to the corresponding element of `upper_bound`.

        Args:
            space_type (SpaceType): space_type
            lower_bound (numpy.ndarray): lower_bound
            upper_bound (numpy.ndarray): upper_bound
        """
        return Space(
            SpaceType.CONTINUOUS,
            lower_bound,
            upper_bound,
        )
