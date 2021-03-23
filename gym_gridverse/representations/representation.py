import abc
from typing import Dict

import numpy as np

from gym_gridverse.agent import Agent
from gym_gridverse.geometry import Orientation
from gym_gridverse.grid import Grid
from gym_gridverse.observation import Observation
from gym_gridverse.representations.spaces import (
    CategoricalSpace,
    ContinuousSpace,
    DiscreteSpace,
    Space,
)
from gym_gridverse.spaces import StateSpace
from gym_gridverse.state import State


class Representation(abc.ABC):
    """Base interface for state, observation and object representation"""

    @property
    @abc.abstractmethod
    def space(self) -> Dict[str, Space]:
        """space of values the representation can return

        Representations convert objects to a dictionary of `str` to
        `np.ndarray` items. The common functionality of these representations
        is to provide space of values that are expected to be returned,
        provided here as a `str` to
        `gym_gridverse.representations.spaces.Space` property.
        """


class StateRepresentation(Representation):
    """Base interface for state representations: enforces `convert`"""

    def __init__(self, state_space: StateSpace):
        if not state_space.can_be_represented:
            raise ValueError(
                'state space contains objects which cannot be represented in state'
            )

        self.state_space = state_space

    @abc.abstractmethod
    def convert(self, s: State) -> Dict[str, np.ndarray]:
        """returns state `s` representation as str -> array dict"""


class ObservationRepresentation(Representation):
    """Base interface for observation representations: enforces `convert`"""

    @abc.abstractmethod
    def convert(self, o: Observation) -> Dict[str, np.ndarray]:
        """returns observation `o` representation as str -> array dict"""


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
        - 'legacy-agent': a 6-value feature representing the agent
          pos/orientation and held object
        - 'legacy-grid': a height x width x 5 shapes array of max values

    Args:
        max_type_index (int): highest value the type of the objects can take
        max_state_index (int): highest value the state of the objects can take
        max_color_value (int): highest value colors can take
        width (int): width of the grid
        height (int): height of the grid

    Returns:
        Dict[str, Space]: {'grid', 'agent', 'item', 'agent_id_grid', 'legacy-agent', 'legacy-grid'}
    """
    assert min([max_type_index, width, height]) > 0, str(
        [max_type_index, width, height]
    )

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

    # Note there is a bug here left intentionally for reproduction purposes:
    # the real space upper bounds are _inclusive_ and thus the max value should
    # not be `height`, but `height - 1`.
    legacy_agent_space = CategoricalSpace(
        np.array(
            [
                height,
                width,
                len(Orientation),
                max_type_index,
                max_state_index,
                max_color_value,
            ]
        )
    )

    legacy_grid_space = CategoricalSpace(
        np.array(
            [[[max_type_index, max_state_index, max_color_value] * 2] * width]
            * height
        ),
    )

    return {
        'grid': grid_space,
        'agent': agent_space,
        'item': item_space,
        'agent_id_grid': agent_id_space,
        'legacy-grid': legacy_grid_space,
        'legacy-agent': legacy_agent_space,
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

    Returns legacy grid and agent representation

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.DefaultStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.DefaultObservationRepresentation`,
    refactored here since DRY

    Args:
        grid (Grid):
        agent (Agent):

    Returns:
        Dict[str, np.ndarray]: {'grid', 'agent', 'item', 'agent_id_grid', 'legacy-agent', 'legacy-grid'}
    """

    item_representation = np.array(
        [agent.obj.type_index, agent.obj.state_index, agent.obj.color.value]
    )

    grid_array_object_channels = np.array(
        [
            [
                [
                    grid[y, x].type_index,
                    grid[y, x].state_index,
                    grid[y, x].color.value,
                ]
                for x in range(grid.shape.width)
            ]
            for y in range(grid.shape.height)
        ]
    )
    grid_agent_position = np.zeros((grid.shape.height, grid.shape.width))
    grid_agent_position[agent.position.astuple()] = 1

    agent_array = np.array([agent.position.y, agent.position.x, 0, 0, 0, 0])
    agent_array[2 + agent.orientation.value] = 1

    # legacy parts
    grid_array_agent_channels = np.zeros(
        (grid.shape.height, grid.shape.width, 3)
    )
    grid_array_agent_channels[agent.position.astuple()] = item_representation

    legacy_agent_array = np.concatenate(
        [
            [agent.position.y, agent.position.x, agent.orientation.value],
            item_representation,
        ]
    )

    return {
        'grid': grid_array_object_channels,
        'agent_id_grid': grid_agent_position,
        'agent': agent_array,
        'item': item_representation,
        'legacy-grid': np.concatenate(
            (grid_array_agent_channels, grid_array_object_channels), axis=-1
        ),
        'legacy-agent': legacy_agent_array,
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

    Categorical data includes 'grid' and 'item' and the legacy values

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
        Dict[str, Space]: {'grid', 'item', 'agent_id_grid', 'agent', 'legacy-agent', 'legacy-grid'}
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

    # legacy work #

    # There is a bug in legacy that remains for reproduction reasons
    # the 'max' values represent the maximum possible value (not the number)
    # so they should be incremented to ensure no overlap
    # this has been properly done in 'grid' and 'item'

    # increment channels to ensure there is no overlap
    no_overlap_legacy_grid_upper_bound = rep['legacy-grid'].upper_bound
    no_overlap_legacy_grid_upper_bound[:, :, [1, 4]] += max_type_index
    no_overlap_legacy_grid_upper_bound[:, :, [2, 5]] += (
        max_type_index + max_state_index
    )
    rep['legacy-grid'] = CategoricalSpace(no_overlap_legacy_grid_upper_bound)

    # default also returns position and orientation, which must be removed
    no_overlap_legay_agent_upper_bound = rep['legacy-agent'].upper_bound
    no_overlap_legay_agent_upper_bound = no_overlap_legay_agent_upper_bound[3:]

    no_overlap_legay_agent_upper_bound[1] += max_type_index
    no_overlap_legay_agent_upper_bound[2] += max_type_index + max_state_index
    rep['legacy-agent'] = CategoricalSpace(no_overlap_legay_agent_upper_bound)

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

    Categorical data includes 'grid' and 'item' and the legacy values

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
        Dict[str, np.ndarray]: {'grid', 'item', 'agent', 'agent_id_grid', 'legacy-agent', 'legacy-grid'}
    """

    rep = default_convert(grid, agent)

    # increment channels to ensure there is no overlap
    rep['grid'][:, :, 1] += max_type_index + 1
    rep['grid'][:, :, 2] += max_type_index + max_state_index + 2

    rep['item'][1] += max_type_index + 1
    rep['item'][2] += max_type_index + max_state_index + 2

    # legacy
    # default also returns item in the legacy grid, which must be removed
    rep['legacy-agent'] = rep['legacy-agent'][3:]

    # This is a bug in legacy that remains for reproduction reasons
    # the 'max' values represent the maximum possible value (not the number)
    # so they should be incremented to ensure no overlap
    # This has been properly done in 'grid' and 'item'
    rep['legacy-grid'][:, :, [1, 4]] += max_type_index
    rep['legacy-grid'][:, :, [2, 5]] += max_type_index + max_state_index
    rep['legacy-agent'][1] += max_type_index
    rep['legacy-agent'][2] += max_type_index + max_state_index

    return rep
