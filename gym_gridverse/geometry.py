from __future__ import annotations

import enum
import itertools as itt
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple, TypeVar

from cached_property import cached_property

T = TypeVar('T')


@dataclass(frozen=True)
class Shape:
    """2D shape, with integer height and width.

    Follows matrix notation:  first index is number of rows, and second index
    is number of columns.
    """

    height: int
    width: int


@dataclass(frozen=True)
class Area:
    """2D area, which extends vertically and horizontally"""

    ys: Tuple[int, int]
    xs: Tuple[int, int]

    def __post_init__(self):
        if self.ys[0] > self.ys[1]:
            raise ValueError('ys ({self.ys}) should be non-decreasing')
        if self.xs[0] > self.xs[1]:
            raise ValueError('xs ({self.xs}) should be non-decreasing')

    @cached_property
    def ymin(self) -> int:
        return min(self.ys)

    @cached_property
    def ymax(self) -> int:
        return max(self.ys)

    @cached_property
    def xmin(self) -> int:
        return min(self.xs)

    @cached_property
    def xmax(self) -> int:
        return max(self.xs)

    @cached_property
    def height(self) -> int:
        return self.ymax - self.ymin + 1

    @cached_property
    def width(self) -> int:
        return self.xmax - self.xmin + 1

    @cached_property
    def top_left(self) -> Position:
        return Position(self.ymin, self.xmin)

    @cached_property
    def top_right(self) -> Position:
        return Position(self.ymin, self.xmax)

    @cached_property
    def bottom_left(self) -> Position:
        return Position(self.ymax, self.xmin)

    @cached_property
    def bottom_right(self) -> Position:
        return Position(self.ymax, self.xmax)

    def y_coordinates(self) -> Iterable[int]:
        """iterator over y coordinates"""
        return range(self.ymin, self.ymax + 1)

    def x_coordinates(self) -> Iterable[int]:
        """iterator over x coordinates"""
        return range(self.xmin, self.xmax + 1)

    def positions(self, selection: str = 'all') -> Iterable[Position]:
        """iterator over all/border/inside positions

        Args:
            selection (str): 'all', 'border', or 'inside'
        Returns:
                Iterable[Position]: selected positions
        """

        if selection not in ['all', 'border', 'inside']:
            raise ValueError(f'invalid selection `{selection}`')

        positions: Iterable[Position]

        if selection == 'all':
            positions = (
                Position(y, x)
                for y in self.y_coordinates()
                for x in self.x_coordinates()
            )

        elif selection == 'border':
            positions = itt.chain(
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

        elif selection == 'inside':
            positions = (
                Position(y, x)
                for y in range(self.ymin + 1, self.ymax)
                for x in range(self.xmin + 1, self.xmax)
            )

        return positions

    def contains(self, position: Position) -> bool:
        return (
            self.ymin <= position.y <= self.ymax
            and self.xmin <= position.x <= self.xmax
        )


@dataclass(frozen=True)
class Position:
    """2D position (y, x), with `y` extending downward and `x` extending rightward"""

    y: int
    x: int

    def to_tuple(self) -> Tuple[int, int]:
        return self.y, self.x

    @staticmethod
    def from_orientation(orientation: Orientation) -> Position:
        try:
            return _position_from_orientation[orientation]
        except KeyError as error:
            raise TypeError(
                'method expectes orientation {orientation}'
            ) from error

    def __add__(self, other: T) -> T:
        if isinstance(other, Position):
            return Position(self.y + other.y, self.x + other.x)

        if isinstance(other, Area):
            return Area(
                (self.y + other.ymin, self.y + other.ymax),
                (self.x + other.xmin, self.x + other.xmax),
            )

        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: Position) -> Position:
        try:
            return Position(self.y - other.y, self.x - other.x)
        except AttributeError:
            return NotImplemented

    def __neg__(self) -> Position:
        return Position(-self.y, -self.x)

    @staticmethod
    def manhattan_distance(p: Position, q: Position) -> float:
        diff = p - q
        return abs(diff.y) + abs(diff.x)

    @staticmethod
    def euclidean_distance(p: Position, q: Position) -> float:
        diff = p - q
        return math.sqrt(diff.y ** 2 + diff.x ** 2)


class Orientation(enum.Enum):
    FORWARD = 0
    BACKWARD = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()

    def __mul__(self, other: T) -> T:
        if isinstance(other, Orientation):
            return _orientation_rotations[self, other]

        if isinstance(other, Position):
            if self is Orientation.FORWARD:
                return Position(other.y, other.x)

            if self is Orientation.BACKWARD:
                return Position(-other.y, -other.x)

            if self is Orientation.RIGHT:
                return Position(other.x, -other.y)

            if self is Orientation.LEFT:
                return Position(-other.x, other.y)

            assert False

        if isinstance(other, Area):
            if self is Orientation.FORWARD:
                return Area(
                    (other.ymin, other.ymax),
                    (other.xmin, other.xmax),
                )

            if self is Orientation.BACKWARD:
                return Area(
                    (-other.ymax, -other.ymin),
                    (-other.xmax, -other.xmin),
                )

            if self is Orientation.RIGHT:
                return Area(
                    (other.xmin, other.xmax),
                    (-other.ymax, -other.ymin),
                )

            if self is Orientation.LEFT:
                return Area(
                    (-other.xmax, -other.xmin),
                    (other.ymin, other.ymax),
                )

        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self) -> Orientation:
        return _orientation_neg[self]

    def as_radians(self) -> float:
        # TODO: test
        radians = {
            Orientation.FORWARD: 0.0,
            Orientation.LEFT: math.pi / 2,
            Orientation.BACKWARD: math.pi,
            Orientation.RIGHT: math.pi * 3 / 2,
        }

        return radians[self]


@dataclass(unsafe_hash=True)
class Transform:
    """A grid-based rigid body transformation, also a ``pose`` (position and orientation)"""

    position: Position
    orientation: Orientation

    def __mul__(self, other: T) -> T:
        """Returns rigid body transformation of other geometric objects

        Implements the following behaviors:
        1. transform * transform -> transformed transform
        2. transform * position -> transformed position
        3. transform * orientation -> transformed orientation
        4. transform * area -> transformed area
        """

        if isinstance(other, Transform):
            return Transform(
                self.position + self.orientation * other.position,
                self.orientation * other.orientation,
            )

        if isinstance(other, (Position, Area)):
            return self.position + self.orientation * other

        if isinstance(other, Orientation):
            return self.orientation * other

        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self) -> Transform:
        return Transform(
            -(-self.orientation * self.position),
            -self.orientation,
        )


def get_manhattan_boundary(position: Position, distance: int) -> List[Position]:
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
        position (Position): The center of the return boundary (excluded)
        distance (int): The distance of the boundary returned

    Returns:
        List[Position]: List of positions (excluding pos) representing the boundary
    """
    if distance <= 0:
        raise ValueError(f'distance ({distance}) must be positive')

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


def distance_function_factory(name: str):
    # TODO: test
    if name == 'manhattan':
        return Position.manhattan_distance

    if name == 'euclidean':
        return Position.euclidean_distance

    raise ValueError(f'invalid distance function name {name}')


class StrideDirection(enum.Enum):
    NW = enum.auto()
    NE = enum.auto()


def diagonal_strides(
    area: Area, stride_direction: StrideDirection
) -> Iterator[Position]:

    if stride_direction == StrideDirection.NW:
        positions = (
            Position(area.ymax - stride + k, area.xmax - k)
            for stride in range(area.height + area.width + 1)
            for k in range(stride + 1)  # stride length
        )

    elif stride_direction == StrideDirection.NE:
        positions = (
            Position(area.ymax - stride + k, area.xmin + k)
            for stride in range(area.height + area.width + 1)
            for k in range(stride + 1)  # stride length
        )

    else:
        raise NotImplementedError

    yield from filter(area.contains, positions)


# cached values (used to avoid if-else chains)

# for Position.from_orientation
_position_from_orientation = {
    Orientation.FORWARD: Position(-1, 0),
    Orientation.RIGHT: Position(0, 1),
    Orientation.BACKWARD: Position(1, 0),
    Orientation.LEFT: Position(0, -1),
}


# for Orientation.__mul__
_orientation_rotations = {
    (Orientation.FORWARD, Orientation.FORWARD): Orientation.FORWARD,
    (Orientation.FORWARD, Orientation.RIGHT): Orientation.RIGHT,
    (Orientation.FORWARD, Orientation.BACKWARD): Orientation.BACKWARD,
    (Orientation.FORWARD, Orientation.LEFT): Orientation.LEFT,
    #
    (Orientation.RIGHT, Orientation.FORWARD): Orientation.RIGHT,
    (Orientation.RIGHT, Orientation.RIGHT): Orientation.BACKWARD,
    (Orientation.RIGHT, Orientation.BACKWARD): Orientation.LEFT,
    (Orientation.RIGHT, Orientation.LEFT): Orientation.FORWARD,
    #
    (Orientation.BACKWARD, Orientation.FORWARD): Orientation.BACKWARD,
    (Orientation.BACKWARD, Orientation.RIGHT): Orientation.LEFT,
    (Orientation.BACKWARD, Orientation.BACKWARD): Orientation.FORWARD,
    (Orientation.BACKWARD, Orientation.LEFT): Orientation.RIGHT,
    #
    (Orientation.LEFT, Orientation.FORWARD): Orientation.LEFT,
    (Orientation.LEFT, Orientation.RIGHT): Orientation.FORWARD,
    (Orientation.LEFT, Orientation.BACKWARD): Orientation.RIGHT,
    (Orientation.LEFT, Orientation.LEFT): Orientation.BACKWARD,
}

# for Orientation.neg
_orientation_neg = {
    Orientation.FORWARD: Orientation.FORWARD,
    Orientation.RIGHT: Orientation.LEFT,
    Orientation.BACKWARD: Orientation.BACKWARD,
    Orientation.LEFT: Orientation.RIGHT,
}
