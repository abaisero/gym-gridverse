from __future__ import annotations

from typing import List, Set, Tuple, Type, Union, cast

from .geometry import Area, Orientation, Position, Shape
from .grid_object import Floor, GridObject, GridObjectFactory, Hidden


class Grid:
    """A two-dimensional grid of objects.

    A container of :py:class:`~gym_gridverse.grid_object.GridObject`. This is
    typically used to represent either the global state of the environment, or
    a partial agent view.  This is basically a two-dimensional array, with some
    additional methods which help in querying and manipulating its objects.
    """

    def __init__(self, objects: List[List[GridObject]]):
        """Constructs a grid from the given grid-objects

        Args:
            objects (List[List[~gym_gridverse.grid_object.GridObject]]): grid of GridObjects
        """
        self.objects = objects
        self.shape = Shape(len(objects), len(objects[0]))
        self.area = Area((0, self.shape.height - 1), (0, self.shape.width - 1))

    @staticmethod
    def from_shape(
        shape: Union[Shape, Tuple[int, int]],
        *,
        factory: GridObjectFactory = Floor,
    ) -> Grid:
        """Constructs a grid with the given shape, with objects generated from the factory.

        Args:
            shape (Union[~gym_gridverse.geometry.Shape, Tuple[int, int]]):
            factory (~gym_gridverse.grid_object.GridObjectFactory):
        Returns:
            Grid: The grid of the appropriate size, with generated objects
        """

        try:
            shape = cast(Shape, shape)
            height, width = shape.height, shape.width
        except AttributeError:
            shape = cast(Tuple[int, int], shape)
            height, width = shape
        objects = [[factory() for _ in range(width)] for _ in range(height)]
        return Grid(objects)

    def __eq__(self, other) -> bool:
        try:
            return self.shape == other.shape and all(
                self[position] == other[position]
                for position in self.area.positions()
            )
        except AttributeError:
            return NotImplemented

    def object_types(self) -> Set[Type[GridObject]]:
        """Returns the set of object types in the grid

        Returns:
            Set[Type[GridObject]]:
        """
        return set(type(self[position]) for position in self.area.positions())

    def get(
        self,
        position: Union[Position, Tuple[int, int]],
        *,
        factory: GridObjectFactory,
    ) -> GridObject:
        """Gets the grid object in the position, or generates one from the factory.

        Args:
            position (Union[~gym_gridverse.geometry.Position, Tuple[int, int]]):
            factory (~gym_gridverse.grid_object.GridObjectFactory):
        Returns:
            GridObject:
        """
        try:
            return self[position]
        except IndexError:
            return factory()

    def __getitem__(
        self, position: Union[Position, Tuple[int, int]]
    ) -> GridObject:
        try:
            position = cast(Position, position)
            y, x = position.yx
        except AttributeError:
            position = cast(Tuple[int, int], position)
            y, x = position

        return self.objects[y][x]

    def __setitem__(
        self, position: Union[Position, Tuple[int, int]], obj: GridObject
    ):
        try:
            position = cast(Position, position)
            y, x = position.yx
        except AttributeError:
            position = cast(Tuple[int, int], position)
            y, x = position

        if not isinstance(obj, GridObject):
            raise TypeError('grid can only contain grid objects')

        self.objects[y][x] = obj

    def swap(self, p: Position, q: Position):
        """Swaps the grid objects at two positions.

        Args:
            p (~gym_gridverse.geometry.Position):
            q (~gym_gridverse.geometry.Position):
        """
        self[p], self[q] = self[q], self[p]

    def subgrid(self, area: Area) -> Grid:
        """Returns subgrid slice at given area.

        Cells included in the area but outside of the grid are represented as
        Hidden objects.

        Args:
            area (~gym_gridverse.geometry.Area): The area to be sliced
        Returns:
            Grid: New instance, sliced appropriately
        """

        return Grid(
            [
                [
                    self.objects[y][x]
                    if 0 <= y < self.area.height and 0 <= x < self.area.width
                    else Hidden()
                    for x in area.x_coordinates()
                ]
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
            orientation (~gym_gridverse.geometry.Orientation): The rotation orientation
        Returns:
            Grid: New instance rotated appropriately
        """
        try:
            rotation_function = _grid_rotation_functions[other]
        except KeyError:
            return NotImplemented
        else:
            objects = rotation_function(self.objects)
            return Grid(objects)

    __rmul__ = __mul__

    def __hash__(self):
        return hash(tuple(map(tuple, self.objects)))

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.shape.height}x{self.shape.width} objects={self.objects}>'


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
