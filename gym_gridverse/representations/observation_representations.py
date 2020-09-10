from typing import Dict

import numpy as np

from gym_gridverse.observation import Observation
from gym_gridverse.representations.representation import (
    ObservationRepresentation,
    default_convert,
    default_representation_space,
    no_overlap_convert,
    no_overlap_representation_space,
)
from gym_gridverse.spaces import ObservationSpace


class DefaultObservationRepresentation(ObservationRepresentation):
    """The default representation for observations

    Simply returns the observation as indices. See
    `gym_gridverse.representations.representation.default_representation_space`
    and `gym_gridverse.representations.representation.default_convert` for more
    information
    """

    def __init__(self, observation_space: ObservationSpace):
        self.observation_space = observation_space

    @property
    def space(self) -> Dict[str, np.ndarray]:
        max_type_index = self.observation_space.max_grid_object_type
        max_state_index = self.observation_space.max_grid_object_status
        max_color_value = self.observation_space.max_object_color

        return default_representation_space(
            max_type_index,
            max_state_index,
            max_color_value,
            self.observation_space.grid_shape.width,
            self.observation_space.grid_shape.height,
        )

    def convert(self, o: Observation) -> Dict[str, np.ndarray]:
        if not self.observation_space.contains(o):
            raise ValueError('Input observation not contained in space')

        return default_convert(o.grid, o.agent)


class NoOverlapObservationRepresentation(ObservationRepresentation):
    """Representation that ensures that the numbers represent unique things

    Simply returns the observation as indices, except that channels do not
    overlap. See
    `gym_gridverse.representations.representation.no_overlap_representation_space`
    and `gym_gridverse.representations.representation.no_overlap_convert` for
    more information
    """

    def __init__(self, observation_space: ObservationSpace):
        self.observation_space = observation_space

    @property
    def space(self) -> Dict[str, np.ndarray]:
        max_type_index = self.observation_space.max_grid_object_type
        max_state_index = self.observation_space.max_grid_object_status
        max_color_value = self.observation_space.max_object_color

        return no_overlap_representation_space(
            max_type_index,
            max_state_index,
            max_color_value,
            self.observation_space.grid_shape.width,
            self.observation_space.grid_shape.height,
        )

    def convert(self, o: Observation) -> Dict[str, np.ndarray]:
        if not self.observation_space.contains(o):
            raise ValueError('Input observation not contained in space')

        max_type_index = self.observation_space.max_grid_object_type
        max_state_index = self.observation_space.max_grid_object_status

        return no_overlap_convert(
            o.grid, o.agent, max_type_index, max_state_index
        )


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
