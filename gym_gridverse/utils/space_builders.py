from typing import Optional, Sequence, Type

from gym_gridverse.debugging import checkraise
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
        checkraise(
            lambda: self.grid_shape is not None,
            RuntimeError,
            'StateSpaceBuilder.build() called without setting `grid_shape`',
        )
        checkraise(
            lambda: self.object_types is not None,
            RuntimeError,
            'StateSpaceBuilder.build() called without setting `object_types`',
        )
        checkraise(
            lambda: self.colors is not None,
            RuntimeError,
            'StateSpaceBuilder.build() called without setting `colors`',
        )

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
        checkraise(
            lambda: self.grid_shape is not None,
            RuntimeError,
            'ObservationSpaceBuilder.build() called without setting `grid_shape`',
        )
        checkraise(
            lambda: self.object_types is not None,
            RuntimeError,
            'ObservationSpaceBuilder.build() called without setting `object_types`',
        )
        checkraise(
            lambda: self.colors is not None,
            RuntimeError,
            'ObservationSpaceBuilder.build() called without setting `colors`',
        )

        return ObservationSpace(self.grid_shape, self.object_types, self.colors)
