from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np

from gym_gridverse.debugging import checkraise

from .geometry import Area, Orientation, Position, Shape
from .grid_object import Floor, GridObject, Hidden

PositionOrTuple = Union[Position, Tuple[int, int]]


class Grid:
    """The state of the environment (minus the agent): a two-dimensional board of objects

    A container of :py:class:`~gym_gridverse.grid_object.GridObject`. This is
    basically a two-dimensional array, with some additional functions to
    simplify interacting with the objects, such as getting areas
    """

    def __init__(
        self, height: int, width: int, *, objects: Optional[np.ndarray] = None
    ):
        """Constructs a `height` x `width` grid of :py:class:`~gym_gridverse.grid_object.Floor`

        Args:
            height (int):
            width (int):

        """
        # TODO: improve __init__ interface;  we basically never want to
        # implicitly default to creating a grid of Floor tiles
        if objects is None:
            objects = np.array(
                [[Floor() for _ in range(width)] for _ in range(height)]
            )

        self.shape = Shape(height, width)
        self._grid = objects

    @property
    def height(self):
        return self.shape.height

    @property
    def width(self):
        return self.shape.width

    @staticmethod
    def from_objects(objects: Sequence[Sequence[GridObject]]) -> Grid:
        """constructor from matrix of GridObjects

        Args:
            objects (Sequence[Sequence[GridObject]]): initialized grid objects
        Returns:
            Grid: Grid containing those objects
        """
        # verifies input is shaped as a matrix
        objects_array = np.asarray(objects)
        return Grid(*objects_array.shape, objects=objects_array)

    def to_objects(self) -> List[List[GridObject]]:
        return self._grid.tolist()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented

        if self.shape != other.shape:
            return False

        for pos in self.positions():
            if self[pos] != other[pos]:
                return False

        return True

    @property
    def area(self) -> Area:
        return Area((0, self.height - 1), (0, self.width - 1))

    # TODO: remove;  Grid is not a collection of positions
    def __contains__(self, position: Position) -> bool:
        """checks if position is in the grid"""
        y, x = position
        return 0 <= y < self.height and 0 <= x < self.width

    def positions(self) -> Iterable[Position]:
        """iterator over positions"""
        return self.area.positions()

    def positions_border(self) -> Iterable[Position]:
        """iterator over border positions"""
        return self.area.positions_border()

    def positions_inside(self) -> Iterable[Position]:
        """iterator over inside positions"""
        return self.area.positions_inside()

    def get_position(self, x: GridObject) -> Position:
        for position in self.positions():
            if self[position] is x:
                return position

        raise ValueError(f'GridObject {x} not found')

    def object_types(self) -> Set[Type[GridObject]]:
        """returns object types currently in the grid"""
        return set(type(self[position]) for position in self.positions())

    def __getitem__(self, position: PositionOrTuple) -> GridObject:
        y, x = (
            (position.y, position.x)
            if isinstance(position, Position)
            else position
        )

        try:
            return self._grid[y, x]
        except IndexError as e:
            # TODO: test
            raise IndexError(f'position {position} not in grid') from e

    def __setitem__(self, position: PositionOrTuple, obj: GridObject):
        checkraise(
            lambda: isinstance(obj, GridObject),
            TypeError,
            'grid can only contain entities',
        )

        y, x = (
            (position.y, position.x)
            if isinstance(position, Position)
            else position
        )

        try:
            self._grid[y, x] = obj
        except IndexError as e:
            # TODO: test
            raise IndexError(f'position {position} not in grid') from e

    def swap(self, p: Position, q: Position):
        """swap the objects at two positions"""
        checkraise(lambda: p in self, ValueError, 'position {} not in grid', p)
        checkraise(lambda: q in self, ValueError, 'position {} not in grid', q)
        self[p], self[q] = self[q], self[p]

    def subgrid(self, area: Area) -> Grid:
        """returns grid sliced at a given area

        Cells included in the area but outside of the grid are represented as
        Hidden objects.

        Args:
            area (Area): The area to be sliced
        Returns:
            Grid: New instance, sliced appropriately
        """

        def get_obj(position: Position) -> GridObject:
            return self[position] if position in self else Hidden()

        return Grid.from_objects(
            [
                [get_obj(Position(y, x)) for x in area.positions_xs()]
                for y in area.positions_ys()
            ]
        )

    def change_orientation(self, orientation: Orientation) -> Grid:
        """returns grid as seen from someone facing the given direction

        E.g. for orientation E, the grid

        AB
        CD

        becomes

        BD
        AC

        Args:
            orientation (Orientation): The orientation of the viewer
        Returns:
            Grid: New instance rotated appropriately
        """
        times = {
            Orientation.N: 0,
            Orientation.S: 2,
            Orientation.E: 1,
            Orientation.W: 3,
        }
        objects = np.rot90(self._grid, times[orientation])
        return Grid.from_objects(objects)

    def __hash__(self):
        return hash(tuple(map(tuple, self._grid.tolist())))

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.height}x{self.width} objects={self.to_objects()!r}>'
