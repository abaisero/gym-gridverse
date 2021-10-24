from typing import Dict

import numpy as np

from gym_gridverse.debugging import gv_debug
from gym_gridverse.observation import Observation
from gym_gridverse.representations.representation import (
    ObservationRepresentation,
    default_convert,
    default_representation_space,
    no_overlap_convert,
    no_overlap_representation_space,
)
from gym_gridverse.representations.spaces import Space
from gym_gridverse.spaces import ObservationSpace


class DefaultObservationRepresentation(ObservationRepresentation):
    """The default representation for observations

    Simply returns the observation as indices. See
    :class:`~gym_gridverse.representations.representation.default_representation_space`
    and :class:`~gym_gridverse.representations.representation.default_convert`
    for more information

    Note that the observation does not include the position or orientation of
    the agent in the agent aspect
    """

    def __init__(self, observation_space: ObservationSpace):
        self.observation_space = observation_space

    @property
    def space(self) -> Dict[str, Space]:
        max_type_index = self.observation_space.max_grid_object_type
        max_state_index = self.observation_space.max_grid_object_status
        max_color_value = self.observation_space.max_object_color

        space = default_representation_space(
            max_type_index,
            max_state_index,
            max_color_value,
            self.observation_space.grid_shape.width,
            self.observation_space.grid_shape.height,
        )

        # observation does not include state information about the agent
        del space['agent']

        return space

    def convert(self, o: Observation) -> Dict:
        if gv_debug() and not self.observation_space.contains(o):
            raise ValueError('observation_space does not contain observation')

        conversion = default_convert(o.grid, o.agent)

        # observation does not include state information about the agent
        del conversion['agent']

        return conversion


class NoOverlapObservationRepresentation(ObservationRepresentation):
    """Representation that ensures that the numbers represent unique things

    Simply returns the observation as indices, except that channels do not
    overlap. See
    :func:`~gym_gridverse.representations.representation.no_overlap_representation_space`
    and
    :func:`~gym_gridverse.representations.representation.no_overlap_convert`
    for more information

    """

    # TODO: test

    def __init__(self, observation_space: ObservationSpace):
        self.observation_space = observation_space

    @property
    def space(self) -> Dict[str, Space]:
        max_type_index = self.observation_space.max_grid_object_type
        max_state_index = self.observation_space.max_grid_object_status
        max_color_value = self.observation_space.max_object_color

        space = no_overlap_representation_space(
            max_type_index,
            max_state_index,
            max_color_value,
            self.observation_space.grid_shape.width,
            self.observation_space.grid_shape.height,
        )

        # agent state variables are not observable
        del space['agent']

        return space

    def convert(self, o: Observation) -> Dict[str, np.ndarray]:
        if gv_debug() and not self.observation_space.contains(o):
            raise ValueError('observation_space does not contain observation')

        max_type_index = self.observation_space.max_grid_object_type
        max_state_index = self.observation_space.max_grid_object_status

        conversion = no_overlap_convert(
            o.grid, o.agent, max_type_index, max_state_index
        )

        del conversion['agent']

        return conversion


class CompactObservationRepresentation(ObservationRepresentation):
    """Returns observations as indices but 'not sparse'

    Will jump over unused indices to allow for smaller spaces

    TODO: implement

    """


def create_observation_representation(
    name: str, observation_space: ObservationSpace
) -> ObservationRepresentation:
    """Factory function for observation representations

    Returns:
        Representation:
    """
    # TODO: test

    if name == 'default':
        return DefaultObservationRepresentation(observation_space)

    if name == 'no_overlap':
        return NoOverlapObservationRepresentation(observation_space)

    if name == 'compact':
        raise NotImplementedError

    raise ValueError(f'invalid name {name}')
