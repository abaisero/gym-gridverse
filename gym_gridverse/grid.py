from __future__ import annotations

from copy import deepcopy
from typing import Iterable, List, Sequence, Set, Type

import numpy as np

from .geometry import Area, Orientation, Position, PositionOrTuple, Shape
from .grid_object import Floor, GridObject, Hidden


class Grid:
    """The state of the environment (minus the agent): a two-dimensional board of objects

    A container of :py:class:`~gym_gridverse.grid_object.GridObject`. This is
    basically a two-dimensional array, with some additional functions to
    simplify interacting with the objects, such as getting areas
    """

    def __init__(self, height: int, width: int):
        """Constructs a `height` x `width` grid of :py:class:`~gym_gridverse.grid_object.Floor`

        Args:
            height (int):
            width (int):

        """
        self.shape = Shape(height, width)
        self._grid = np.array(
            [[Floor() for _ in range(width)] for _ in range(height)]
        )

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
        array = np.array(objects)

        grid = Grid(*array.shape)
        for pos in grid.positions():
            grid[pos] = array[pos.y, pos.x]

        return grid

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

    # TODO remove;  Grid is not a collection of positions
    def __contains__(self, position: PositionOrTuple) -> bool:
        """checks if position is in the grid"""
        position = Position.from_position_or_tuple(position)
        return 0 <= position.y < self.height and 0 <= position.x < self.width

    def _check_contains(
        self,
        position: PositionOrTuple,
        exception_type: Type[Exception] = ValueError,
    ):
        """raises value error if position is not in the grid"""
        position = Position.from_position_or_tuple(position)
        if position not in self:
            raise exception_type(f'Position {position} ')

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
        position = Position.from_position_or_tuple(position)

        if position not in self:
            raise IndexError(f'position {position} not in grid')

        return self._grid[position.y, position.x]

    def __setitem__(self, position: PositionOrTuple, obj: GridObject):
        position = Position.from_position_or_tuple(position)

        if position not in self:
            raise IndexError(f'position {position} not in grid')

        if not isinstance(obj, GridObject):
            TypeError('grid can only contain entities')

        y, x = position
        self._grid[y, x] = obj

    def swap(self, p: Position, q: Position):
        """swap the objects at two positions"""
        if p not in self:
            raise ValueError(f'position {p} not in grid')

        if q not in self:
            raise ValueError(f'position {q} not in grid')

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
        subgrid = Grid(area.height, area.width)
        grid_copy = deepcopy(self)
        for pos_to in subgrid.positions():
            pos_from = pos_to + area.top_left

            try:
                obj = grid_copy[pos_from]
            except IndexError:
                obj = Hidden()

            subgrid[pos_to] = obj
            # subgrid[pos_to] = (
            #     np_grid[pos_from.y, pos_from.x]
            #     if pos_from in self
            #     else Hidden()
            # )

        return subgrid

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
        objects = np.rot90(self._grid, times[orientation]).tolist()
        objects = deepcopy(objects)
        return Grid.from_objects(objects)

    def __hash__(self):
        return hash(tuple(map(tuple, self._grid.tolist())))

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.height}x{self.width} objects={self.to_objects()!r}>'
