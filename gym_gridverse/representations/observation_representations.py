from typing import Dict

import numpy as np

from gym_gridverse.grid_object import NoneGridObject
from gym_gridverse.observation import Observation
from gym_gridverse.representations.representation import Representation
from gym_gridverse.spaces import ObservationSpace


class DefaultObservationRepresentation(Representation):
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

    def convert(self, observation: Observation) -> Dict[str, np.ndarray]:
        # TODO either ignore pylint or avoid specialization of signature input

        # TODO make sure input observation fits space?

        agent_obj_array = np.array(
            [
                observation.agent.obj.type_index,
                observation.agent.obj.state_index,
                observation.agent.obj.color.value,
            ]
        )

        grid_array_object_channels = np.array(
            [
                [
                    [
                        observation.grid[y, x].type_index,
                        observation.grid[y, x].state_index,
                        observation.grid[y, x].color.value,
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
        grid_array_agent_channels[observation.agent.position] = agent_obj_array

        grid_array = np.concatenate(
            (grid_array_agent_channels, grid_array_object_channels), axis=-1
        )
        return {'grid': grid_array, 'agent': agent_obj_array}


class CompactObservationRepresentation(Representation):
    """Returns observations as indices but 'not sparse'

    Will jump over unused indices to allow for smaller spaces

    """

    def __init__(self):
        pass

    @property
    def space(self) -> Dict[str, np.ndarray]:
        pass

    def convert(self, x) -> Dict[str, np.ndarray]:
        pass


def create_observation_representation() -> Representation:
    """Factory function for observation representations

    TODO: nyi, current returns `ObservationToArray`

    Returns:
        Representation: [TODO:description]
    """
    return DefaultObservationRepresentation()
