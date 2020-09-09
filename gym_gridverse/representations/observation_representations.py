from typing import Dict

import numpy as np
from gym_gridverse.grid_object import NoneGridObject
from gym_gridverse.observation import Observation
from gym_gridverse.representations.representation import \
    ObservationRepresentation
from gym_gridverse.spaces import ObservationSpace


class DefaultObservationRepresentation(ObservationRepresentation):
    """The default representation for observations

    Simply returns the observation as indices

    """

    def __init__(self, observation_space: ObservationSpace):
        self.observation_space = observation_space

    @property
    def space(self) -> Dict[str, np.ndarray]:
        max_type_index = max(
            object_type.type_index
            for object_type in self.observation_space.object_types
        )
        max_state_index = max(
            object_type.num_states()
            for object_type in self.observation_space.object_types
        )
        max_color_value = max(
            color.value for color in self.observation_space.colors
        )

        grid_array = np.array(
            [
                [
                    [max_type_index, max_state_index, max_color_value]
                    for x in range(self.observation_space.grid_shape.width)
                ]
                for y in range(self.observation_space.grid_shape.height)
            ]
        )
        agent_array = np.array(
            [max_type_index, max_state_index, max_color_value]
        )
        return {'grid': grid_array, 'agent': agent_array}

    def convert(self, o: Observation) -> Dict[str, np.ndarray]:

        # TODO make sure input observation fits space?

        agent_obj_array = np.array(
            [
                o.agent.obj.type_index,
                o.agent.obj.state_index,
                o.agent.obj.color.value,
            ]
        )

        grid_array_object_channels = np.array(
            [
                [
                    [
                        o.grid[y, x].type_index,
                        o.grid[y, x].state_index,
                        o.grid[y, x].color.value,
                    ]
                    for x in range(self.observation_space.grid_shape.width)
                ]
                for y in range(self.observation_space.grid_shape.height)
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
                    for x in range(self.observation_space.grid_shape.width)
                ]
                for y in range(self.observation_space.grid_shape.height)
            ]
        )
        grid_array_agent_channels[o.agent.position] = agent_obj_array

        grid_array = np.concatenate(
            (grid_array_agent_channels, grid_array_object_channels), axis=-1
        )
        return {'grid': grid_array, 'agent': agent_obj_array}


class CompactObservationRepresentation(ObservationRepresentation):
    """Returns observations as indices but 'not sparse'

    Will jump over unused indices to allow for smaller spaces

    TODO: implement

    """

    def __init__(self):
        pass

    @property
    def space(self) -> Dict[str, np.ndarray]:
        pass

    def convert(self, o: Observation) -> Dict[str, np.ndarray]:
        pass


def create_observation_representation(
    observation_space: ObservationSpace,
) -> ObservationRepresentation:
    """Factory function for observation representations

    TODO: nyi, current returns `DefaultObservationRepresentation`

    Returns:
        Representation:
    """
    return DefaultObservationRepresentation(observation_space)
