from __future__ import annotations

import enum
import math
from typing import Callable, List, NamedTuple, Tuple


class Shape(NamedTuple):
    """
    2D shape which follow matrix notation: first index indicates number of
    rows, and second index indicates number of columns.
    """

    height: int
    width: int


class Area:
    def __init__(self, y: Tuple[int, int], x: Tuple[int, int]):
        self.ymin, self.ymax = min(y), max(y)
        self.xmin, self.xmax = min(x), max(x)

    @property
    def height(self):
        return self.ymax - self.ymin + 1

    @property
    def width(self):
        return self.xmax - self.xmin + 1

    @property
    def top_left(self):
        return Position(self.ymin, self.xmin)

    @property
    def top_right(self):
        return Position(self.ymin, self.xmax)

    @property
    def bottom_left(self):
        return Position(self.ymax, self.xmin)

    @property
    def bottom_right(self):
        return Position(self.ymax, self.xmax)

    def contains(self, position: Position):
        return (
            self.ymin <= position.y <= self.ymax
            and self.xmin <= position.x <= self.xmax
        )

    def __hash__(self):
        return hash(((self.ymin, self.ymax), (self.xmin, self.xmax)))

    def __eq__(self, other):
        if not isinstance(other, Area):
            return NotImplemented

        return (
            self.ymin == other.ymin
            and self.ymax == other.ymax
            and self.xmin == other.xmin
            and self.xmax == other.xmax
        )

    def __repr__(self):
        return f'Area(({self.ymin}, {self.ymax}), ({self.xmin}, {self.xmax}))'


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

    @staticmethod
    def subtract(p1, p2):
        return Position(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def manhattan_distance(p1, p2) -> float:
        diff = _2D_Point.subtract(p1, p2)
        return abs(diff.y) + abs(diff.x)

    @staticmethod
    def euclidean_distance(p1, p2) -> float:
        diff = _2D_Point.subtract(p1, p2)
        return math.sqrt(diff.y ** 2 + diff.x ** 2)


# using inheritance to allow checking semantic types with isinstance without
# aliasing Position with DeltaPosition


class Position(_2D_Point):
    pass


class DeltaPosition(_2D_Point):
    def rotate(self, orientation: Orientation) -> DeltaPosition:
        if orientation is Orientation.N:
            rotated_dpos = DeltaPosition(self.y, self.x)

        elif orientation is Orientation.S:
            rotated_dpos = DeltaPosition(-self.y, -self.x)

        elif orientation is Orientation.E:
            rotated_dpos = DeltaPosition(self.x, -self.y)

        elif orientation is Orientation.W:
            rotated_dpos = DeltaPosition(-self.x, self.y)

        else:
            assert False

        return rotated_dpos


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


DistanceFunction = Callable[[Position, Position], float]
