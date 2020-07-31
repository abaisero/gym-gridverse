from __future__ import annotations

import enum
from typing import NamedTuple


class Shape(NamedTuple):
    """
    2D shape which follow matrix notation: first index indicates number of
    rows, and second index indicates number of columns.
    """

    height: int
    width: int


class _2D_Point(NamedTuple):
    """
    2D coordinates which follow matrix notation: first index indicates rows
    from top to bottom, and second index indicates columns from left to right.
    """

    y: int
    x: int


# using inheritance to allow checking semantic types with isinstance without
# aliasing Position with DeltaPosition


class Position(_2D_Point):
    pass


class DeltaPosition(_2D_Point):
    pass


class Orientation(enum.Enum):
    N = 0
    S = enum.auto()
    E = enum.auto()
    W = enum.auto()

    def as_delta_position(self, dist: int = 1) -> DeltaPosition:
        if self is Orientation.N:
            delta_position = DeltaPosition(-dist, 0)

        elif self is Orientation.S:
            delta_position = DeltaPosition(dist, 0)

        elif self is Orientation.E:
            delta_position = DeltaPosition(0, dist)

        elif self is Orientation.W:
            delta_position = DeltaPosition(0, -dist)

        else:
            assert False

        return delta_position
