from __future__ import annotations

from typing import List, Optional, Set, Tuple, Type, Union

import numpy as np
from numpy.typing import ArrayLike

from .geometry import Area, Orientation, Position, Shape
from .grid_object import Floor, GridObject, GridObjectFactory, Hidden

PositionOrTuple = Union[Position, Tuple[int, int]]
ShapeOrTuple = Union[Shape, Tuple[int, int]]


class Grid:
    """The state of the environment (minus the agent): a two-dimensional board of objects

    A container of :py:class:`~gym_gridverse.grid_object.GridObject`. This is
    basically a two-dimensional array, with some additional functions to
    simplify interacting with the objects, such as getting areas
    """

    def __init__(self, objects: ArrayLike):
        """Constructs a `height` x `width` grid of :py:class:`~gym_gridverse.grid_object.Floor`

        Args:
            objects (numpy.typing.ArrayLike): grid of GridObjects
        """
        objects = np.asarray(objects)

        self._objects = objects
        self._shape = Shape(*objects.shape)
        self._area = Area((0, self.shape.height - 1), (0, self.shape.width - 1))

    @staticmethod
    def from_shape(
        shape: ShapeOrTuple,
        *,
        factory: Optional[GridObjectFactory] = None,
    ) -> Grid:
        if factory is None:
            factory = Floor

        height, width = (
            (shape.height, shape.width) if isinstance(shape, Shape) else shape
        )
        objects = [[factory() for _ in range(width)] for _ in range(height)]
        return Grid(np.asarray(objects))

    def to_list(self) -> List[List[GridObject]]:
        return self._objects.tolist()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented

        if self.shape != other.shape:
            return False

        for pos in self.area.positions():
            if self[pos] != other[pos]:
                return False

        return True

    @property
    def shape(self):
        return self._shape

    @property
    def area(self) -> Area:
        return self._area

    def get_position(self, x: GridObject) -> Position:
        for position in self.area.positions():
            if self[position] is x:
                return position

        raise ValueError(f'GridObject {x} not found')

    def object_types(self) -> Set[Type[GridObject]]:
        """returns object types currently in the grid"""
        return set(type(self[position]) for position in self.area.positions())

    def get(self, position: PositionOrTuple, factory: GridObjectFactory):
        try:
            return self[position]
        except IndexError:
            return factory()

    def __getitem__(self, position: PositionOrTuple) -> GridObject:
        y, x = (
            (position.y, position.x)
            if isinstance(position, Position)
            else position
        )

        try:
            return self._objects[y, x]
        except IndexError as e:
            # TODO: test
            raise IndexError(f'position {position} not in grid') from e

    def __setitem__(self, position: PositionOrTuple, obj: GridObject):
        if not isinstance(obj, GridObject):
            raise TypeError('grid can only contain grid objects')

        y, x = (
            (position.y, position.x)
            if isinstance(position, Position)
            else position
        )

        try:
            self._objects[y, x] = obj
        except IndexError as e:
            # TODO: test
            raise IndexError(f'position {position} not in grid') from e

    def swap(self, p: Position, q: Position):
        """swap the objects at two positions"""
        if not self.area.contains(p):
            raise ValueError(f'Position {p} not in grid area')
        if not self.area.contains(p):
            raise ValueError(f'Position {q} not in grid area')

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

        return Grid(
            [
                [self.get((y, x), Hidden) for x in area.x_coordinates()]
                for y in area.y_coordinates()
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
        objects = np.rot90(self._objects, times[orientation])
        return Grid(objects)

    def __hash__(self):
        return hash(tuple(map(tuple, self._objects.tolist())))

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.shape.height}x{self.shape.width} objects={self.to_list()!r}>'
