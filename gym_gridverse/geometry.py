from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, NamedTuple


class Shape(NamedTuple):
    """
    2D shape which follow matrix notation: first index indicates number of
    rows, and second index indicates number of columns.
    """

    height: int
    width: int


class Area(NamedTuple):
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def height(self):
        return self.y1 - self.y0 + 1

    @property
    def width(self):
        return self.x1 - self.x0 + 1


class _2D_Point(NamedTuple):
    """
    2D coordinates which follow matrix notation: first index indicates rows
    from top to bottom, and second index indicates columns from left to right.
    """

    y: int
    x: int

    @staticmethod
    def add(p1, p2):
        return Position(p1[0] + p2[0], p1[1] + p2[1])


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

    def rotate_left(self):
        rotations = {
            self.N: self.W,
            self.W: self.S,
            self.S: self.E,
            self.E: self.N,
        }

        return rotations[self]

    def rotate_right(self):
        rotations = {
            self.N: self.E,
            self.E: self.S,
            self.S: self.W,
            self.W: self.N,
        }

        return rotations[self]


def get_manhattan_boundary(pos: Position, distance: int) -> List[Position]:
    """Returns the cells (excluding pos) with Manhattan distance of pos

    For distance = 1, will return the left, upper, right and lower cell of
    position. For longer distances, the extended boundary is returned:

    E.g. for distance = 2 the cells denoted by 'x' are returned:

      x
     x x
    x . x
     x x
      x

    Args:
        pos (Position): The center of the return boundary (excluded)
        distance (int): The distance of the boundary returned

    Returns:
        List[Position]: List of positions (excluding pos) representing the boundary
    """
    assert distance > 0

    boundary: List[Position] = []
    # from top, adding points clockwise in 4 straight lines
    boundary.extend(
        Position(pos.y - distance + i, pos.x + i) for i in range(distance)
    )
    boundary.extend(
        Position(pos.y + i, pos.x + distance - i) for i in range(distance)
    )
    boundary.extend(
        Position(pos.y + distance - i, pos.x - i) for i in range(distance)
    )
    boundary.extend(
        Position(pos.y - i, pos.x - distance + i) for i in range(distance)
    )
    return boundary
