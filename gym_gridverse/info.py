from __future__ import annotations

from copy import deepcopy
from typing import Iterator, Optional, Sequence

import numpy as np

from .geometry import Area, Orientation, Position, Shape
from .grid_object import Floor, GridObject, Hidden, NoneGridObject


class Grid:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self._grid = np.array(
            [[Floor() for _ in range(width)] for _ in range(height)],
        )

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
            grid[pos] = array[pos]

        return grid

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
    def shape(self) -> Shape:
        return Shape(self.height, self.width)

    def __contains__(self, position: Position) -> bool:
        """checks if position is in the grid"""
        return 0 <= position.y < self.height and 0 <= position.x < self.width

    def _check_contains(self, position: Position):
        """raises value error if position is not in the grid"""
        if position not in self:
            raise ValueError(f'Position {position} ')

    def positions(self) -> Iterator[Position]:
        for indices in iter(np.indices(self.shape).reshape(2, -1).T):
            yield Position(*indices)

    def get_position(self, x: GridObject) -> Position:
        for position in self.positions():
            if self[position] is x:
                return position

        raise ValueError(f'GridObject {x} not found')

    def __getitem__(self, position: Position) -> GridObject:
        return self._grid[position]

    def __setitem__(self, position: Position, obj: GridObject):
        if not isinstance(obj, GridObject):
            TypeError('grid can only contain entities')
        self._grid[position] = obj

    def swap(self, p: Position, q: Position):
        """swap the objects at two positions"""
        self._check_contains(p)
        self._check_contains(q)
        self[p], self[q] = self[q], self[p]

    def subgrid(self, area: Area) -> Grid:
        """returns grid sliced at a given area

        Cells included in the area but outside of the grid are represented as
        Hidden objects.

        Args:
            area (Area): The area to be sliced
        Returnd:
            Grid: New instance, sliced appropriately
        """
        subgrid = Grid(area.height, area.width)

        for pos_to in subgrid.positions():
            pos_from = Position(pos_to.y + area.y0, pos_to.x + area.x0)
            subgrid[pos_to] = (
                deepcopy(self[pos_from]) if pos_from in self else Hidden()
            )

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


class Agent:
    def __init__(
        self,
        position: Position,
        orientation: Orientation,
        obj: Optional[GridObject] = None,
    ):
        if obj is None:
            obj = NoneGridObject()

        self.position = position
        self.orientation = orientation
        self.obj: GridObject = obj

    def get_pov_area(self) -> Area:
        # TODO generalize to arbitrary rectangle centered at arbitrary position
        # TODO make area_size an environment parameter
        area_size = 7

        y0, x0 = self.position

        if self.orientation is Orientation.N:
            y0 -= area_size - 1
            x0 -= area_size // 2

        elif self.orientation is Orientation.S:
            x0 -= area_size // 2

        elif self.orientation is Orientation.E:
            y0 -= area_size // 2

        elif self.orientation is Orientation.W:
            y0 -= area_size // 2
            x0 -= area_size - 1

        else:
            assert False

        y1 = y0 + area_size - 1
        x1 = x0 + area_size - 1
        return Area(y0, x0, y1, x1)
