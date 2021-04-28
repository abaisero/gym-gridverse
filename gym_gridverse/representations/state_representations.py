from typing import Dict

import numpy as np

from gym_gridverse.debugging import checkraise
from gym_gridverse.representations.representation import (
    StateRepresentation,
    default_convert,
    default_representation_space,
    no_overlap_convert,
    no_overlap_representation_space,
)
from gym_gridverse.representations.spaces import Space
from gym_gridverse.spaces import StateSpace
from gym_gridverse.state import State


class DefaultStateRepresentation(StateRepresentation):
    """The default representation for state

    Simply returns the state as indices. See
    :func:`gym_gridverse.representations.representation.default_representation_space`
    and :func:`gym_gridverse.representations.representation.default_convert`
    for more information

    """

    @property
    def space(self) -> Dict[str, Space]:
        max_type_index = self.state_space.max_agent_object_type
        max_state_index = self.state_space.max_grid_object_status
        max_color_value = self.state_space.max_object_color

        return default_representation_space(
            max_type_index,
            max_state_index,
            max_color_value,
            self.state_space.grid_shape.width,
            self.state_space.grid_shape.height,
        )

    def convert(self, s: State) -> Dict[str, np.ndarray]:
        checkraise(
            lambda: self.state_space.contains(s),
            ValueError,
            'Input state not contained in space',
        )

        return default_convert(s.grid, s.agent)


class NoOverlapStateRepresentation(StateRepresentation):
    """Representation that ensures that the numbers represent unique things

    Simply returns the state as indices, except that channels do not
    overlap. See
    `gym_gridverse.representations.representation.no_overlap_representation_space`
    and `gym_gridverse.representations.representation.no_overlap_convert` for
    more information
    """

    @property
    def space(self) -> Dict[str, np.ndarray]:
        max_type_index = self.state_space.max_grid_object_type
        max_state_index = self.state_space.max_grid_object_status
        max_color_value = self.state_space.max_object_color

        return no_overlap_representation_space(
            max_type_index,
            max_state_index,
            max_color_value,
            self.state_space.grid_shape.width,
            self.state_space.grid_shape.height,
        )

    def convert(self, s: State) -> Dict[str, np.ndarray]:
        checkraise(
            lambda: self.state_space.contains(s),
            ValueError,
            'Input state not contained in space',
        )

        max_type_index = self.state_space.max_grid_object_type
        max_state_index = self.state_space.max_grid_object_status

        return no_overlap_convert(
            s.grid, s.agent, max_type_index, max_state_index
        )


class CompactStateRepresentation(StateRepresentation):
    """Returns state as indices but 'not sparse'

    Will jump over unused indices to allow for smaller spaces

    TODO: implement

    """

    def __init__(self, state_space: StateSpace):
        super().__init__(state_space)
        raise NotImplementedError


def create_state_representation(
    name: str, state_space: StateSpace
) -> StateRepresentation:
    """Factory function for state representations

    Returns:
        Representation:
    """

    if name == 'default':
        return DefaultStateRepresentation(state_space)

    if name == 'no_overlap':
        return NoOverlapStateRepresentation(state_space)

    if name == 'compact':
        raise NotImplementedError

    raise ValueError(f'invalid name {name}')
