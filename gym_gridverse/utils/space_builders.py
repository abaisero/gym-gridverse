from typing import Optional, Sequence, Type

from gym_gridverse.geometry import Shape
from gym_gridverse.grid_object import Color, GridObject
from gym_gridverse.spaces import ObservationSpace, StateSpace


class StateSpaceBuilder:
    def __init__(self):
        self.grid_shape: Optional[Shape] = None
        self.object_types: Optional[Sequence[Type[GridObject]]] = None
        self.colors: Optional[Sequence[Color]] = None

    def set_grid_shape(self, grid_shape: Shape):
        self.grid_shape = grid_shape

    def set_object_types(self, object_types: Sequence[Type[GridObject]]):
        self.object_types = object_types

    def set_colors(self, colors: Sequence[Color]):
        self.colors = colors

    def build(self) -> StateSpace:
        if self.grid_shape is None:
            raise RuntimeError('`grid_shape` was not set')

        if self.object_types is None:
            raise RuntimeError('`object_types` was not set')

        if self.colors is None:
            raise RuntimeError('`colors` was not set')

        return StateSpace(self.grid_shape, self.object_types, self.colors)


class ObservationSpaceBuilder:
    def __init__(self):
        self.grid_shape: Optional[Shape] = None
        self.object_types: Optional[Sequence[Type[GridObject]]] = None
        self.colors: Optional[Sequence[Color]] = None

    def set_grid_shape(self, grid_shape: Shape):
        self.grid_shape = grid_shape

    def set_object_types(self, object_types: Sequence[Type[GridObject]]):
        self.object_types = object_types

    def set_colors(self, colors: Sequence[Color]):
        self.colors = colors

    def build(self) -> ObservationSpace:
        if self.grid_shape is None:
            raise RuntimeError('`grid_shape` was not set')

        if self.object_types is None:
            raise RuntimeError('`object_types` was not set')

        if self.colors is None:
            raise RuntimeError('`colors` was not set')

        return ObservationSpace(self.grid_shape, self.object_types, self.colors)
