from __future__ import annotations

import enum
import itertools as itt
import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, Union, overload


@dataclass(frozen=True)
class Shape:
    """2D shape, with integer height and width.

    Follows matrix notation:  first index is number of rows, and second index
    is number of columns.
    """

    height: int
    width: int

    @property
    def as_tuple(self) -> Tuple[int, int]:
        return self.height, self.width


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

    @staticmethod
    def from_positions(positions: Sequence[Position]) -> Area:
        ys, xs = zip(*(position.yx for position in positions))
        return Area((min(ys), max(ys)), (min(xs), max(xs)))

    @property
    def ymin(self) -> int:
        return self.ys[0]

    @property
    def ymax(self) -> int:
        return self.ys[1]

    @property
    def xmin(self) -> int:
        return self.xs[0]

    @property
    def xmax(self) -> int:
        return self.xs[1]

    @property
    def height(self) -> int:
        return self.ymax - self.ymin + 1

    @property
    def width(self) -> int:
        return self.xmax - self.xmin + 1

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

    @property
    def yx(self) -> Tuple[int, int]:
        return self.y, self.x

    @staticmethod
    def from_orientation(orientation: Orientation) -> Position:
        try:
            return _position_from_orientation[orientation]
        except KeyError as error:
            raise TypeError(
                'method expectes orientation {orientation}'
            ) from error

    @overload
    def __add__(self, other: Position) -> Position:
        ...

    @overload
    def __add__(self, other: Area) -> Area:
        ...

    def __add__(self, other: Union[Position, Area]) -> Union[Position, Area]:
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
        return math.sqrt(diff.y**2 + diff.x**2)


class Orientation(enum.Enum):
    FORWARD = 0
    BACKWARD = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()

    # aliases
    F = FORWARD
    B = BACKWARD
    L = LEFT
    R = RIGHT

    @overload
    def __mul__(self, other: Orientation) -> Orientation:
        ...

    @overload
    def __mul__(self, other: Position) -> Position:
        ...

    @overload
    def __mul__(self, other: Area) -> Area:
        ...

    def __mul__(
        self, other: Union[Orientation, Position, Area]
    ) -> Union[Orientation, Position, Area]:
        if isinstance(other, Orientation):
            return _orientation_rotations[self, other]

        if isinstance(other, Position):
            if self is Orientation.F:
                return Position(other.y, other.x)

            if self is Orientation.B:
                return Position(-other.y, -other.x)

            if self is Orientation.R:
                return Position(other.x, -other.y)

            if self is Orientation.L:
                return Position(-other.x, other.y)

            assert False

        if isinstance(other, Area):
            if self is Orientation.F:
                return Area(
                    (other.ymin, other.ymax),
                    (other.xmin, other.xmax),
                )

            if self is Orientation.B:
                return Area(
                    (-other.ymax, -other.ymin),
                    (-other.xmax, -other.xmin),
                )

            if self is Orientation.R:
                return Area(
                    (other.xmin, other.xmax),
                    (-other.ymax, -other.ymin),
                )

            if self is Orientation.L:
                return Area(
                    (-other.xmax, -other.xmin),
                    (other.ymin, other.ymax),
                )

        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self) -> Orientation:
        """Negative orientation

        NOTE:  This operator represents the algebraeic notion of negation, with
        FORWARD being the identity element.  That is, -orientation is defined
        such that orientation * -orientation == -orientation * orientation == FORWARD:

            -FORWARD == FORWARD (*not* BACKWARD)
            -BACKWARD == BACKWARD (*not* BACKWARD)
            -LEFT == RIGHT
            -RIGHT == LEFT
        """
        return _orientation_neg[self]


@dataclass(unsafe_hash=True)
class Transform:
    """A grid-based rigid body transformation, also a ``pose`` (position and orientation)"""

    position: Position
    orientation: Orientation

    @overload
    def __mul__(self, other: Transform) -> Transform:
        ...

    @overload
    def __mul__(self, other: Position) -> Position:
        ...

    @overload
    def __mul__(self, other: Area) -> Area:
        ...

    @overload
    def __mul__(self, other: Orientation) -> Orientation:
        ...

    def __mul__(
        self, other: Union[Transform, Position, Area, Orientation]
    ) -> Union[Transform, Position, Area, Orientation]:
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


# cached values (used to avoid if-else chains)

# for Position.from_orientation
_position_from_orientation = {
    Orientation.F: Position(-1, 0),
    Orientation.R: Position(0, 1),
    Orientation.B: Position(1, 0),
    Orientation.L: Position(0, -1),
}


# for Orientation.__mul__
_orientation_rotations = {
    (Orientation.F, Orientation.F): Orientation.F,
    (Orientation.F, Orientation.R): Orientation.R,
    (Orientation.F, Orientation.B): Orientation.B,
    (Orientation.F, Orientation.L): Orientation.L,
    #
    (Orientation.R, Orientation.F): Orientation.R,
    (Orientation.R, Orientation.R): Orientation.B,
    (Orientation.R, Orientation.B): Orientation.L,
    (Orientation.R, Orientation.L): Orientation.F,
    #
    (Orientation.B, Orientation.F): Orientation.B,
    (Orientation.B, Orientation.R): Orientation.L,
    (Orientation.B, Orientation.B): Orientation.F,
    (Orientation.B, Orientation.L): Orientation.R,
    #
    (Orientation.L, Orientation.F): Orientation.L,
    (Orientation.L, Orientation.R): Orientation.F,
    (Orientation.L, Orientation.B): Orientation.R,
    (Orientation.L, Orientation.L): Orientation.B,
}

# for Orientation.neg
_orientation_neg = {
    Orientation.F: Orientation.F,
    Orientation.R: Orientation.L,
    Orientation.B: Orientation.B,
    Orientation.L: Orientation.R,
}
