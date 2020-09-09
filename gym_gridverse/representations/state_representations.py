from typing import Dict

import numpy as np
from gym_gridverse.grid_object import NoneGridObject
from gym_gridverse.representations.representation import StateRepresentation
from gym_gridverse.spaces import StateSpace
from gym_gridverse.state import State


class DefaultStateRepresentation(StateRepresentation):
    """The default representation for state

    Simply returns the state as indices

    """

    def __init__(self, state_space: StateSpace):
        self.state_space = state_space

    @property
    def space(self) -> Dict[str, np.ndarray]:
        max_type_index = self.state_space.max_agent_object_type
        max_state_index = self.state_space.max_grid_object_status
        max_color_value = self.state_space.max_object_color

        grid_array = np.array(
            [
                [
                    [max_type_index, max_state_index, max_color_value]
                    for x in range(self.state_space.grid_shape.width)
                ]
                for y in range(self.state_space.grid_shape.height)
            ]
        )
        agent_array = np.array(
            [max_type_index, max_state_index, max_color_value]
        )
        return {'grid': grid_array, 'agent': agent_array}

    def convert(self, s: State) -> Dict[str, np.ndarray]:
        if not self.state_space.contains(s):
            import ipdb; ipdb.set_trace()
            raise ValueError('Input state not contained in space')

        agent_obj_array = np.array(
            [
                s.agent.obj.type_index,
                s.agent.obj.state_index,
                s.agent.obj.color.value,
            ]
        )

        grid_array_object_channels = np.array(
            [
                [
                    [
                        s.grid[y, x].type_index,
                        s.grid[y, x].state_index,
                        s.grid[y, x].color.value,
                    ]
                    for x in range(self.state_space.grid_shape.width)
                ]
                for y in range(self.state_space.grid_shape.height)
            ]
        )
        none_grid_object = NoneGridObject()
        grid_array_agent_channels = np.array(
            [
                [
                    [
                        none_grid_object.type_index,  # pylint: disable=no-member
                        none_grid_object.state_index,
                        none_grid_object.color.value,
                    ]
                    for x in range(self.state_space.grid_shape.width)
                ]
                for y in range(self.state_space.grid_shape.height)
            ]
        )
        grid_array_agent_channels[s.agent.position] = agent_obj_array

        grid_array = np.concatenate(
            (grid_array_agent_channels, grid_array_object_channels), axis=-1
        )
        return {'grid': grid_array, 'agent': agent_obj_array}


class CompactStateRepresentation(StateRepresentation):
    """Returns state as indices but 'not sparse'

    Will jump over unused indices to allow for smaller spaces

    TODO: implement

    """

    def __init__(self):
        pass

    @property
    def space(self) -> Dict[str, np.ndarray]:
        pass

    def convert(self, s: State) -> Dict[str, np.ndarray]:
        pass


def create_state_representation(
    state_space: StateSpace,
) -> StateRepresentation:
    """Factory function for state representations

    TODO: nyi, current returns `DefaultStateRepresentation`

    Returns:
        Representation:
    """
    return DefaultStateRepresentation(state_space)
