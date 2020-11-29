from __future__ import annotations

import enum
import itertools as itt
import math
from typing import Callable, Iterator, List, NamedTuple, Tuple, Union


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
    def height(self) -> int:
        return self.ymax - self.ymin + 1

    @property
    def width(self) -> int:
        return self.xmax - self.xmin + 1

    @property
    def top_left(self) -> Position:
        return Position(self.ymin, self.xmin)

    @property
    def top_right(self) -> Position:
        return Position(self.ymin, self.xmax)

    @property
    def bottom_left(self) -> Position:
        return Position(self.ymax, self.xmin)

    @property
    def bottom_right(self) -> Position:
        return Position(self.ymax, self.xmax)

    def positions(self) -> Iterator[Position]:
        """iterator over positions"""

        return (
            Position(y, x)
            for y in range(self.ymin, self.ymax + 1)
            for x in range(self.xmin, self.xmax + 1)
        )

    def positions_border(self) -> Iterator[Position]:
        """iterator over border positions"""

        return itt.chain(
            (
                Position(y, x)
                for y in [self.ymin, self.ymax]
                for x in range(self.xmin, self.xmax + 1)
            ),
            (
                Position(y, x)
                for y in range(self.ymin + 1, self.ymax)
                for x in [self.xmin, self.xmax]
            ),
        )

    def positions_inside(self) -> Iterator[Position]:
        """iterator over inside positions"""

        return (
            Position(y, x)
            for y in range(self.ymin + 1, self.ymax)
            for x in range(self.xmin + 1, self.xmax)
        )

    def contains(self, position: PositionOrTuple) -> bool:
        position = Position.from_position_or_tuple(position)
        return (
            self.ymin <= position.y <= self.ymax
            and self.xmin <= position.x <= self.xmax
        )

    def translate(self, position: PositionOrTuple) -> Area:
        position = Position.from_position_or_tuple(position)
        return Area(
            (self.ymin + position.y, self.ymax + position.y),
            (self.xmin + position.x, self.xmax + position.x),
        )

    def rotate(self, orientation: Orientation) -> Area:
        if orientation is Orientation.N:
            area = Area((self.ymin, self.ymax), (self.xmin, self.xmax))

        elif orientation is Orientation.S:
            area = Area((-self.ymin, -self.ymax), (-self.xmin, -self.xmax))

        elif orientation is Orientation.E:
            area = Area((self.xmin, self.xmax), (-self.ymax, -self.ymin))

        elif orientation is Orientation.W:
            area = Area((-self.xmax, -self.xmin), (self.ymin, self.ymax))

        else:
            assert False

        return area

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
    def add(p1, p2) -> Position:
        return Position(p1[0] + p2[0], p1[1] + p2[1])

    @staticmethod
    def subtract(p1, p2) -> Position:
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
    @staticmethod
    def from_position_or_tuple(position: PositionOrTuple) -> Position:
        return (
            position if isinstance(position, Position) else Position(*position)
        )


PositionOrTuple = Union[Position, Tuple[int, int]]


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
            return DeltaPosition(-dist, 0)

        if self is Orientation.S:
            return DeltaPosition(dist, 0)

        if self is Orientation.E:
            return DeltaPosition(0, dist)

        if self is Orientation.W:
            return DeltaPosition(0, -dist)

        raise RuntimeError

    def as_radians(self) -> float:
        radians = {
            Orientation.N: 0.0,
            Orientation.W: math.pi / 2,
            Orientation.S: math.pi,
            Orientation.E: math.pi * 3 / 2,
        }

        return radians[self]

    def rotate_left(self) -> Orientation:
        rotations = {
            Orientation.N: Orientation.W,
            Orientation.W: Orientation.S,
            Orientation.S: Orientation.E,
            Orientation.E: Orientation.N,
        }

        return rotations[self]

    def rotate_right(self) -> Orientation:
        rotations = {
            Orientation.N: Orientation.E,
            Orientation.E: Orientation.S,
            Orientation.S: Orientation.W,
            Orientation.W: Orientation.N,
        }

        return rotations[self]


def get_manhattan_boundary(
    position: PositionOrTuple, distance: int
) -> List[Position]:
    """Returns the cells (excluding pos) with Manhattan distance of pos

    For distance = 1, will return the left, upper, right and lower cell of
    position. For longer distances, the extended boundary is returned:

    E.g. for distance = 2 the cells denoted by 'x' are returned::

          x
         x x
        x . x
         x x
          x

    Args:
        position (PositionOrTuple): The center of the return boundary (excluded)
        distance (int): The distance of the boundary returned

    Returns:
        List[Position]: List of positions (excluding pos) representing the boundary
    """
    assert distance > 0

    position = Position.from_position_or_tuple(position)

    boundary: List[Position] = []
    # from top, adding points clockwise in 4 straight lines
    boundary.extend(
        Position(position.y - distance + i, position.x + i)
        for i in range(distance)
    )
    boundary.extend(
        Position(position.y + i, position.x + distance - i)
        for i in range(distance)
    )
    boundary.extend(
        Position(position.y + distance - i, position.x - i)
        for i in range(distance)
    )
    boundary.extend(
        Position(position.y - i, position.x - distance + i)
        for i in range(distance)
    )
    return boundary


DistanceFunction = Callable[[Position, Position], float]
