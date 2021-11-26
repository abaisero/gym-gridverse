import abc
from typing import Dict, Tuple

import numpy as np

from gym_gridverse.agent import Agent
from gym_gridverse.grid import Grid
from gym_gridverse.observation import Observation
from gym_gridverse.representations.spaces import (
    CategoricalSpace,
    ContinuousSpace,
    DiscreteSpace,
    Space,
)
from gym_gridverse.spaces import ObservationSpace, StateSpace
from gym_gridverse.state import State


class Representation(abc.ABC):
    """Base interface for state and observation representation

    Representations convert :py:class:`~gym_gridverse.state.State` and
    :py:class:`~gym_gridverse.observation.Observation` objects into
    dictionaries of :py:class:`~numpy.ndarray` values.
    """

    @property
    @abc.abstractmethod
    def space(self) -> Dict[str, Space]:
        """dictionary of represented spaces"""


class StateRepresentation(Representation):
    def __init__(self, state_space: StateSpace):
        if not state_space.can_be_represented:
            raise ValueError(
                'state space contains objects which cannot be represented in state'
            )

        self.state_space = state_space

    @abc.abstractmethod
    def convert(self, state: State) -> Dict[str, np.ndarray]:
        """returns state representation as dictionary of numpy arrays"""


class ObservationRepresentation(Representation):
    def __init__(self, observation_space: ObservationSpace):
        self.observation_space = observation_space

    @abc.abstractmethod
    def convert(self, observation: Observation) -> Dict[str, np.ndarray]:
        """returns observation representation as dictionary of numpy arrays"""


def default_representation_space(
    max_type_index: int,
    max_state_index: int,
    max_color_value: int,
    width: int,
    height: int,
) -> Dict[str, Space]:
    """the naive space of the representation

    Values converted by the representation cannot be outside the space(s)
    returned here

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.DefaultStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.DefaultObservationRepresentation`,
    refactored here since DRY

    Returns a dictionary of
    :py:class:`~gym_gridverse.representations.spaces.Space`, each representing
    either a different aspect or in a different way:

        - 'grid': (height x width x 3) categorical space of the grid of item
          type/status/color
        - 'agent': (y, x, 1, 1, 1, 1) continuous space, where the first 2
          represent the normalized position and the last 4 ones represent the
          one-hot encoding of the orientation
        - 'agent_id_grid': (height x width) one-hot encoding of the agent's location
        - 'item': the categorical item type, status and color (three values)
        - 'agent_id_grid':

    Args:
        max_type_index (int): highest value the type of the objects can take
        max_state_index (int): highest value the state of the objects can take
        max_color_value (int): highest value colors can take
        width (int): width of the grid
        height (int): height of the grid

    Returns:
        Dict[str, Space]: keys {'grid', 'agent', 'item', 'agent_id_grid'}
    """
    if max_type_index < 0:
        raise ValueError(f'negative max_type_index ({max_type_index})')
    if height < 0 or width < 0:
        raise ValueError(f'negative height or width ({height, width})')

    grid_space = CategoricalSpace(
        np.array(
            [[[max_type_index, max_state_index, max_color_value]] * width]
            * height
        )
    )

    # 4 (last) entries for a one-hot encoding of the orientation
    agent_space = ContinuousSpace(
        np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )

    agent_id_space = DiscreteSpace(
        np.zeros((height, width), dtype=int),
        np.ones((height, width), dtype=int),
    )

    item_space = CategoricalSpace(
        np.array(
            [
                max_type_index,
                max_state_index,
                max_color_value,
            ]
        )
    )

    return {
        'grid': grid_space,
        'agent': agent_space,
        'item': item_space,
        'agent_id_grid': agent_id_space,
    }


def default_convert(grid: Grid, agent: Agent) -> Dict[str, np.ndarray]:
    """Default naive convertion of a grid and agent

    Converts grid to a 3 channel (of height x width) representation of:

        - object type index
        - object state index
        - object color index

    e.g. return['grid'][h,w,2] returns the color of object on grid position (h,w)

    Converts agent into a feature values: (y, x, one-hot-encoding of orientation (4 values))

    Converts item into it's three values: (type, status, color)

    Returns 'agent_id_grid', a one-hot encoding of the agent ID (shape of the grid: h x w)

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.DefaultStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.DefaultObservationRepresentation`,
    refactored here since DRY

    Args:
        grid (Grid):
        agent (Agent):

    Returns:
        Dict[str, numpy.ndarray]: keys {'grid', 'agent', 'item', 'agent_id_grid'}
    """

    item_representation = np.array(
        [
            agent.grid_object.type_index(),
            agent.grid_object.state_index,
            agent.grid_object.color.value,
        ]
    )

    # need this to improve efficiency, because python3.7 does not have assignment operators
    def get_obj_array(y: int, x: int) -> Tuple[int, int, int]:
        obj = grid[y, x]
        return obj.type_index(), obj.state_index, obj.color.value

    grid_array_object_channels = np.array(
        [
            [get_obj_array(y, x) for x in range(grid.shape.width)]
            for y in range(grid.shape.height)
        ],
        int,
    )
    grid_agent_position = np.zeros((grid.shape.height, grid.shape.width), int)
    grid_agent_position[agent.position.y, agent.position.x] = 1

    agent_array = np.zeros(6)
    agent_array[0] = (2 * agent.position.y - grid.shape.height + 1) / (
        grid.shape.height - 1
    )
    agent_array[1] = (2 * agent.position.x - grid.shape.width + 1) / (
        grid.shape.width - 1
    )
    agent_array[2 + agent.orientation.value] = 1

    return {
        'grid': grid_array_object_channels,
        'agent_id_grid': grid_agent_position,
        'agent': agent_array,
        'item': item_representation,
    }


def no_overlap_representation_space(
    max_type_index: int,
    max_state_index: int,
    max_color_value: int,
    width: int,
    height: int,
) -> Dict[str, Space]:
    """A representation space where categorical data does not overlap

    Returns the same shape and info as :func:`default_representation_space` but
    ensures that
    :class:`~gym_gridverse.representations.spaces.CategoricalSpace` data have
    no overlap. That means that each value maps uniquely to a construct, even
    accross different returned items. As a result, the size of the spaces are
    generally larger.  The
    :class:`~gym_gridverse.representations.spaces.DiscreteSpace` values are
    left unmodified.

    Categorical data includes 'grid' and 'item'

    Will return a dictionary that contains 'grid' and 'item':

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.NoOverlapStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.NoOverlapObservationRepresentation`,
    refactored here since DRY

    Args:
        max_type_index (int): highest value the type of the objects can take
        max_state_index (int): highest value the state of the objects can take
        max_color_value (int): highest value colors can take
        width (int): width of the grid
        height (int): height of the grid

    Returns:
        Dict[str, Space]: keys {'grid', 'item', 'agent_id_grid', 'agent'}
    """

    rep = default_representation_space(
        max_type_index, max_state_index, max_color_value, width, height
    )

    # increment channels to ensure there is no overlap
    no_overlap_grid_upper_bound = rep['grid'].upper_bound
    no_overlap_grid_upper_bound[:, :, 1] += max_type_index + 1
    no_overlap_grid_upper_bound[:, :, 2] += max_type_index + max_state_index + 2

    # default also returns agent ids as fourth channel, which must be removed
    no_overlap_grid_upper_bound = no_overlap_grid_upper_bound[:, :, :3]
    rep['grid'] = CategoricalSpace(no_overlap_grid_upper_bound)

    item_upper_bound = rep['item'].upper_bound
    item_upper_bound[1] += max_type_index + 1
    item_upper_bound[2] += max_type_index + max_state_index + 2
    rep['item'] = CategoricalSpace(item_upper_bound)

    return rep


def no_overlap_convert(
    grid: Grid, agent: Agent, max_type_index: int, max_state_index: int
) -> Dict[str, np.ndarray]:
    """Converts to a representation without overlapping categorical data

    Returns the same shape and info as :func:`default_convert` but ensures that
    categorical data have no overlap. That means that each value maps uniquely
    to a construct, even accross different returned items. As a result, the
    size of the spaces are generally larger.  The discrete data are left
    unmodified.

    Categorical data includes 'grid' and 'item' values

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.NoOverlapStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.NoOverlapObservationRepresentation`,
    refactored here since DRY


    Args:
        max_type_index (int): highest value the type of the objects can take
        max_state_index (int): highest value the state of the objects can take
        max_color_value (int): highest value colors can take
        width (int): width of the grid
        height (int): height of the grid

    Returns:
        Dict[str, numpy.ndarray]: keys {'grid', 'item', 'agent', 'agent_id_grid'}
    """

    rep = default_convert(grid, agent)

    # increment channels to ensure there is no overlap
    rep['grid'][:, :, 1] += max_type_index + 1
    rep['grid'][:, :, 2] += max_type_index + max_state_index + 2

    rep['item'][1] += max_type_index + 1
    rep['item'][2] += max_type_index + max_state_index + 2

    return rep
