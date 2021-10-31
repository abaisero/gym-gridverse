from __future__ import annotations

import enum
import itertools as itt
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Tuple

from cached_property import cached_property


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

    def translate(self, position: Position) -> Area:
        return Area(
            (self.ymin + position.y, self.ymax + position.y),
            (self.xmin + position.x, self.xmax + position.x),
        )

    def rotate(self, orientation: Orientation) -> Area:
        if orientation is Orientation.N:
            return Area((self.ymin, self.ymax), (self.xmin, self.xmax))

        if orientation is Orientation.S:
            return Area((-self.ymax, -self.ymin), (-self.xmax, -self.xmin))

        if orientation is Orientation.E:
            return Area((self.xmin, self.xmax), (-self.ymax, -self.ymin))

        if orientation is Orientation.W:
            return Area((-self.xmax, -self.xmin), (self.ymin, self.ymax))

        raise TypeError('orientation must be of type Orientation')


@dataclass(frozen=True)
class Position:
    """2D position (y, x), with `y` extending downward and `x` extending rightward"""

    y: int
    x: int

    def to_tuple(self) -> Tuple[int, int]:
        return self.y, self.x

    def __add__(self, other: Position) -> Position:
        try:
            return Position(self.y + other.y, self.x + other.x)
        except AttributeError:
            return NotImplemented

    def __sub__(self, other: Position) -> Position:
        try:
            return Position(self.y - other.y, self.x - other.x)
        except AttributeError:
            return NotImplemented

    def __neg__(self) -> Position:
        return Position(-self.y, -self.x)

    def rotate(self, orientation: Orientation) -> Position:
        if orientation is Orientation.N:
            return Position(self.y, self.x)

        if orientation is Orientation.S:
            return Position(-self.y, -self.x)

        if orientation is Orientation.E:
            return Position(self.x, -self.y)

        if orientation is Orientation.W:
            return Position(-self.x, self.y)

        raise TypeError('orientation must be of type Orientation')

    @staticmethod
    def manhattan_distance(p: Position, q: Position) -> float:
        diff = p - q
        return abs(diff.y) + abs(diff.x)

    @staticmethod
    def euclidean_distance(p: Position, q: Position) -> float:
        diff = p - q
        return math.sqrt(diff.y ** 2 + diff.x ** 2)


class Orientation(enum.Enum):
    N = 0
    S = enum.auto()
    E = enum.auto()
    W = enum.auto()

    def as_position(self, dist: int = 1) -> Position:
        if self is Orientation.N:
            return Position(-dist, 0)

        if self is Orientation.S:
            return Position(dist, 0)

        if self is Orientation.E:
            return Position(0, dist)

        if self is Orientation.W:
            return Position(0, -dist)

        assert False, 'self must be of type Orientation'

    def as_radians(self) -> float:
        # TODO: test
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

    def rotate_back(self) -> Orientation:
        rotations = {
            Orientation.N: Orientation.S,
            Orientation.E: Orientation.W,
            Orientation.S: Orientation.N,
            Orientation.W: Orientation.E,
        }

        return rotations[self]


@dataclass(unsafe_hash=True)
class Pose:
    """Pair of position and orientation"""

    position: Position
    orientation: Orientation

    def absolute_position(self, relative_position: Position) -> Position:
        """get the absolute position from a delta position relative to the pose"""
        return self.position + relative_position.rotate(self.orientation)

    def front_position(self) -> Position:
        """get the position in front of the pose"""
        return self.absolute_position(Orientation.N.as_position())

    def absolute_area(self, relative_area: Area) -> Area:
        """gets absolute area corresponding to given relative area

        The relative ares is relative to the agent's POV, with position (0, 0)
        representing the agent's position.  The absolute area is the relative
        ares translated and rotated such as to indicate the agent's POV in
        absolute state coordinates.
        """
        return relative_area.rotate(self.orientation).translate(self.position)


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
