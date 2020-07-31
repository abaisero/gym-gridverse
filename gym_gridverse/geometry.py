import enum
from typing import NamedTuple


class Coordinates_2D(NamedTuple):
    """
    2D coordinates which follow matrix notation: first index indicates rows
    from top to bottom, and second index indicates columns from left to right.
    """

    y: int
    x: int


# using inheritance to allow checking semantic types with isinstance without
# aliasing Position with DeltaPosition


class Position(Coordinates_2D):
    pass


class DeltaPosition(Coordinates_2D):
    pass


class Orientation(enum.Enum):
    N = 0
    S = enum.auto()
    E = enum.auto()
    W = enum.auto()
