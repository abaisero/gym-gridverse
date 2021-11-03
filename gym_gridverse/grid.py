from __future__ import annotations

from typing import List, Set, Tuple, Type, Union

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

    def __init__(self, objects: List[List[GridObject]]):
        """Constructs a `height` x `width` grid of :py:class:`~gym_gridverse.grid_object.Floor`

        Args:
            objects (List[List[GridObject]]): grid of GridObjects
        """
        self._objects = objects
        self.shape = Shape(len(objects), len(objects[0]))
        self.area = Area((0, self.shape.height - 1), (0, self.shape.width - 1))

    @staticmethod
    def from_shape(
        shape: ShapeOrTuple,
        *,
        factory: GridObjectFactory = Floor,
    ) -> Grid:
        height, width = (
            (shape.height, shape.width) if isinstance(shape, Shape) else shape
        )
        objects = [[factory() for _ in range(width)] for _ in range(height)]
        return Grid(objects)

    def to_list(self) -> List[List[GridObject]]:
        return self._objects

    def __eq__(self, other) -> bool:
        try:
            return self.shape == other.shape and all(
                self[position] == other[position]
                for position in self.area.positions()
            )
        except AttributeError:
            return NotImplemented

    def object_types(self) -> Set[Type[GridObject]]:
        """returns object types currently in the grid"""
        return set(type(self[position]) for position in self.area.positions())

    def get(self, position: PositionOrTuple, factory: GridObjectFactory):
        try:
            return self[position]
        except IndexError:
            return factory()

    def __getitem__(self, position: PositionOrTuple) -> GridObject:
        y, x = self._validate_position(position)
        return self._objects[y][x]

    def __setitem__(self, position: PositionOrTuple, obj: GridObject):
        y, x = self._validate_position(position)

        if not isinstance(obj, GridObject):
            raise TypeError('grid can only contain grid objects')

        self._objects[y][x] = obj

    def _validate_position(self, position: PositionOrTuple) -> Tuple[int, int]:
        try:
            y, x = position.to_tuple()
        except AttributeError:
            y, x = position

        if not (0 <= y < self.shape.height and 0 <= x < self.shape.width):
            raise IndexError(f'position {(y,x)} not in grid')

        return y, x

    def swap(self, p: Position, q: Position):
        self[p], self[q] = self[q], self[p]

    def subgrid(self, area: Area) -> Grid:
        """returns subgrid slice at given area

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

    def __mul__(self, other: Orientation) -> Grid:
        """returns grid transformed according to given orientation.

        NOTE:  this product follows rigid body transform conventions, whereby
        the orientation represents a transform between frames to apply on the
        grid, e.g.,

        orientation * grid = expected
        -----------   ----   --------

                      ABC    CFI
              RIGHT * DEF  = BEH
                      GHI    ADG


        becomes

        BD
        AC

        Args:
            orientation (Orientation): The rotation orientation
        Returns:
            Grid: New instance rotated appropriately
        """
        try:
            rotation_function = _grid_rotation_functions[other]
        except KeyError:
            return NotImplemented
        else:
            objects = rotation_function(self._objects)
            return Grid(objects)

    __rmul__ = __mul__

    def __hash__(self):
        return hash(tuple(map(tuple, self.to_list())))

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.shape.height}x{self.shape.width} objects={self.to_list()!r}>'


def _rotate_matrix_forward(data):
    return data


def _rotate_matrix_right(data):
    return [list(row) for row in zip(*data[::-1])]


def _rotate_matrix_left(data):
    return [list(row) for row in zip(*data)][::-1]


def _rotate_matrix_backward(data):
    return [d[::-1] for d in data[::-1]]


# for Grid.__mul__
_grid_rotation_functions = {
    Orientation.F: _rotate_matrix_forward,
    Orientation.R: _rotate_matrix_left,
    Orientation.B: _rotate_matrix_backward,
    Orientation.L: _rotate_matrix_right,
}
