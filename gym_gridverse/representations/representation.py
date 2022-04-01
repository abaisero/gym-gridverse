import abc
from typing import Dict, Generic, Set, Type, TypeVar

import numpy as np

from gym_gridverse.grid_object import Color, GridObject
from gym_gridverse.observation import Observation
from gym_gridverse.representations.spaces import Space
from gym_gridverse.spaces import ObservationSpace, StateSpace
from gym_gridverse.state import State


class StateRepresentation:
    """Converts a :py:class:`~gym_gridverse.state.State` into a dictionary of :py:class:`~numpy.ndarray`."""

    def __init__(self, state_space: StateSpace):
        if not state_space.can_be_represented:
            raise ValueError(
                'state space contains objects which cannot be represented in state'
            )

        self.state_space = state_space

    @property
    @abc.abstractmethod
    def space(self) -> Dict[str, Space]:
        """returns representation space as as dictionary of numpy arrays"""
        assert False

    @abc.abstractmethod
    def convert(self, state: State) -> Dict[str, np.ndarray]:
        """returns state representation as dictionary of numpy arrays"""
        assert False


class ObservationRepresentation:
    """Converts a :py:class:`~gym_gridverse.observation.Observation` into a dictionary of :py:class:`~numpy.ndarray`."""

    def __init__(self, observation_space: ObservationSpace):
        self.observation_space = observation_space

    @property
    @abc.abstractmethod
    def space(self) -> Dict[str, Space]:
        """returns representation space as as dictionary of numpy arrays"""
        assert False

    @abc.abstractmethod
    def convert(self, observation: Observation) -> Dict[str, np.ndarray]:
        """returns observation representation as dictionary of numpy arrays"""
        assert False


T = TypeVar('T', State, Observation, GridObject)


class ArrayRepresentation(Generic[T]):
    @property
    @abc.abstractmethod
    def space(self) -> Space:
        assert False

    @abc.abstractmethod
    def convert(self, obj: T) -> np.ndarray:
        assert False


# grid-object representations


def default_grid_object_representation_space(
    grid_object_types: Set[Type[GridObject]],
    grid_object_colors: Set[Color],
) -> Space:
    """The default space of the representation

    Returns a :py:class:`~gym_gridverse.representations.spaces.Space`
    representing the space of a grid-object represented using type-index,
    status-index, and color-index.

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.DefaultGridObjectStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.DefaultGridObjectObservationRepresentation`,
    refactored here because of DRY.
    """
    max_agent_object_type_index = max(
        grid_object_type.type_index() for grid_object_type in grid_object_types
    )
    # TODO minor bug:  the max state index is -1 compared to the num-states
    max_agent_object_state_index = max(
        grid_object_type.num_states() for grid_object_type in grid_object_types
    )
    max_agent_object_color_index = max(
        color.value for color in grid_object_colors
    )

    return Space.make_categorical_space(
        np.array(
            [
                max_agent_object_type_index,
                max_agent_object_state_index,
                max_agent_object_color_index,
            ]
        )
    )


def default_grid_object_representation_convert(
    grid_object: GridObject,
) -> np.ndarray:
    """The default conversion of a grid-object

    Converts a grid-object into a 3-channel array of:

        - object type index
        - object state index
        - object color index

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.DefaultGridObjectStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.DefaultObservationGridObjectObservationRepresentation`,
    refactored here because of DRY.
    """

    return np.array(
        [
            grid_object.type_index(),
            grid_object.state_index,
            grid_object.color.value,
        ]
    )


def no_overlap_grid_object_representation_space(
    grid_object_types: Set[Type[GridObject]],
    grid_object_colors: Set[Color],
) -> Space:
    """The no-overlap space of the representation

    Returns a :py:class:`~gym_gridverse.representations.spaces.Space`
    representing the space of a grid-object represented using type-index,
    status-index, and color-index.  Guarantees no overlap across channels,
    meaning that each channel uses separate indices.

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.NoOverlapGridObjectStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.NoOverlapGridObjectObservationRepresentation`,
    refactored here because of DRY.
    """
    max_agent_object_type_index = max(
        grid_object_type.type_index() for grid_object_type in grid_object_types
    )
    # TODO minor bug:  the max state index is -1 compared to the num-states
    max_agent_object_state_index = max(
        grid_object_type.num_states() for grid_object_type in grid_object_types
    )
    max_agent_object_color_index = max(
        color.value for color in grid_object_colors
    )

    return Space.make_categorical_space(
        np.array(
            [
                max_agent_object_type_index,
                max_agent_object_type_index + max_agent_object_state_index + 1,
                max_agent_object_type_index
                + max_agent_object_state_index
                + max_agent_object_color_index
                + 2,
            ]
        )
    )


def no_overlap_grid_object_representation_convert(
    grid_object_types: Set[Type[GridObject]],
    grid_object_colors: Set[Color],
    grid_object: GridObject,
) -> np.ndarray:
    """The no-overlap conversion of a grid-object

    Converts a :py:class:`~gym_gridverse.grid_object.GridObject` into a
    3-channel array of type-index, status-index, and color-index.  Guarantees
    no overlap across channels, meaning that each channel uses separate
    indices.

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.NoOverlapGridObjectStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.NoOverlapGridObjectObservationRepresentation`,
    refactored here because of DRY.
    """

    max_agent_object_type_index = max(
        grid_object_type.type_index() for grid_object_type in grid_object_types
    )
    # TODO minor bug:  the max state index is -1 compared to the num-states
    max_agent_object_state_index = max(
        grid_object_type.num_states() for grid_object_type in grid_object_types
    )

    return np.array(
        [
            grid_object.type_index(),
            max_agent_object_type_index + grid_object.state_index + 1,
            max_agent_object_type_index
            + max_agent_object_state_index
            + grid_object.color.value
            + 2,
        ]
    )


def compact_grid_object_representation_space(
    grid_object_type_map: np.ndarray,
    grid_object_state_map: np.ndarray,
    grid_object_color_map: np.ndarray,
) -> Space:
    """The compact space of the representation

    Returns a :py:class:`~gym_gridverse.representations.spaces.Space`
    representing the space of a grid-object represented using type-index,
    status-index, and color-index.  Guarantees a compact no overlap
    representation across channels, meaning that each channel uses separate
    indices, and there are no gaps between the used indices.

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.CompactGridObjectStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.CompactGridObjectObservationRepresentation`,
    refactored here because of DRY.
    """

    return Space.make_categorical_space(
        np.array(
            [
                grid_object_type_map.max(),
                grid_object_state_map.max(),
                grid_object_color_map.max(),
            ]
        )
    )


def compact_grid_object_representation_convert(
    grid_object_type_map: np.ndarray,
    grid_object_state_map: np.ndarray,
    grid_object_color_map: np.ndarray,
    grid_object: GridObject,
) -> np.ndarray:
    """The no-overlap conversion of a grid-object

    Converts a :py:class:`~gym_gridverse.grid_object.GridObject` into a
    3-channel array of type-index, status-index, and color-index.  Guarantees a
    compact no overlap representation across channels, meaning that each
    channel uses separate indices, and there are no gaps between the used
    indices.

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.CompactGridObjectStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.CompactGridObjectObservationRepresentation`,
    refactored here because of DRY.
    """

    i = grid_object.type_index()
    j = grid_object.state_index
    k = grid_object.color.value
    return np.array(
        [
            grid_object_type_map[i],
            grid_object_state_map[i, j],
            grid_object_color_map[k],
        ]
    )
