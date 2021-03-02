import abc
from typing import Dict

import numpy as np

from gym_gridverse.agent import Agent
from gym_gridverse.geometry import Orientation
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import NoneGridObject
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import StateSpace
from gym_gridverse.state import State


class Representation(metaclass=abc.ABCMeta):
    """Base interface for state, observation and object representation"""

    @property
    @abc.abstractmethod
    def space(self) -> Dict[str, np.ndarray]:
        """range of values the representation can return

        returns a string -> numpy array of max values, e.g: same shape as
        `convert` but returns max values. So `self.convert() <= self.space`,
        note the greater _or equal_
        """


class StateRepresentation(Representation, metaclass=abc.ABCMeta):
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


class ObservationRepresentation(Representation, metaclass=abc.ABCMeta):
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
) -> Dict[str, np.ndarray]:
    """the naive space of the representation, returns max indices each value can take

    Values cannot be _greater than_ those returned here (there is a `<=`
    relationship)

    NOTE: used by `DefaultStateRepresentation` and
    `DefaultObservationRepresentation`, refactored here since DRY

    Returns a dictionary of numpy arrays, each representing either a different aspect or in a different way:

        - 'grid': (height x width x 4) grid of max item type/status/color and
          agent position
        - 'agent': (max y, max x, 1, 1, 1, 1), where the last 4 ones represent
          the one-hot encoding of the orientation
        - 'item': the max item type, status and color (three values)
        - 'legacy-agent': a 6-value feature array representing the agent
          pos/orientation and held object
        - 'legacy-grid': a height x width x 5 shapes array of max values

    Args:
        max_type_index (int): highest value the type of the objects can take
        max_state_index (int): highest value the state of the objects can take
        max_color_value (int): highest value colors can take
        width (int): width of the grid
        height (int): height of the grid

    Returns:
        Dict[str, np.ndarray]: {'grid', 'agent', 'item', 'legacy-agent', 'legacy-grid'}
    """
    assert min([max_type_index, width, height]) > 0, str(
        [max_type_index, width, height]
    )

    grid_array = np.array(
        [[[max_type_index, max_state_index, max_color_value, 2]] * width]
        * height
    )

    # 4 entries for a one-hot encoding of the orientation
    agent_array = np.array([height - 1, width - 1, 1, 1, 1, 1])

    item_array = np.array(
        [
            max_type_index,
            max_state_index,
            max_color_value,
        ]
    )

    legacy_agent_array = np.array(
        [
            height - 1,
            width - 1,
            len(Orientation) - 1,
            max_type_index,
            max_state_index,
            max_color_value,
        ]
    )

    legacy_grid_array = np.array(
        [[[max_type_index, max_state_index, max_color_value] * 2] * width]
        * height
    )

    return {
        'grid': grid_array,
        'agent': agent_array,
        'item': item_array,
        'legacy-grid': legacy_grid_array,
        'legacy-agent': legacy_agent_array,
    }


def default_convert(grid: Grid, agent: Agent) -> Dict[str, np.ndarray]:
    """Default naive convertion of a grid and agent

    Converts grid to a 4 channel (of height x width) representation of:

        - object type index
        - object state index
        - object color index
        - zeros except for agent's position

    e.g. return['grid'][h,w,2] returns the color of object on grid position (h,w)

    Converts agent into a feature values: (y, x, one-hot-encoding of orientation (4 values))

    Converts holind item into it's three values: (type, status, color)

    Lastly also returns legacy grid and agent representation

    NOTE: used by
    :class:`~gym_gridverse.representations.state_representations.DefaultStateRepresentation`
    and
    :class:`~gym_gridverse.representations.observation_representations.DefaultObservationRepresentation`,
    refactored here since DRY

    Args:
        grid (Grid):
        agent (Agent):

    Returns:
        Dict[str, np.ndarray]: {'grid', 'agent', 'item', 'legacy-agent', 'legacy-grid'}
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
    grid_agent_position = np.zeros((grid.shape.height, grid.shape.width, 1))
    grid_agent_position[agent.position.y, agent.position.x, 0] = 1

    agent_array = np.array([agent.position.y, agent.position.x, 0, 0, 0, 0])
    agent_array[2 + agent.orientation.value] = 1

    # legacy parts
    none_grid_object = NoneGridObject()
    grid_array_agent_channels = np.array(
        [
            [
                [
                    none_grid_object.type_index,  # pylint: disable=no-member
                    none_grid_object.state_index,
                    none_grid_object.color.value,
                ]
                for _ in range(grid.shape.width)
            ]
            for _ in range(grid.shape.height)
        ]
    )
    grid_array_agent_channels[agent.position.astuple()] = item_representation

    legacy_agent_array = np.concatenate(
        [
            [agent.position.y, agent.position.x, agent.orientation.value],
            item_representation,
        ]
    )

    return {
        'grid': np.concatenate(
            (grid_array_object_channels, grid_agent_position), axis=-1
        ),
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
) -> Dict[str, np.ndarray]:
    """similar to the `default_representation_space`, but ensures no overlap between channels

    Values cannot be _greater than_ those returned here (there is a `<=`
    relationship)

    Will return a dictionary that contains 'grid' and 'item':

    return['grid'] is a height x width x 3 shaped array of max item values

        However, where the values in each channel have a unique meaning. As in,
        the values in channel 2 (`grid[h,w,2]`) do not overlap in any of the
        other channels. This is to allow specific index numbers to have unique
        meanings. In short, the values in channel 2 start from the max value of
        channel 1, and the values in channel 3 start from the max of channel 2
        etc.

    return['item'] is a 3-value feature array with max item type/status/color
    channels

        Where the channels reflect those in 'grid'

    Will also return 'legacy-grid' and 'legacy-agent', containing values used
    by old code bases

    NOTE: used by `NoOverlapStateRepresentation` and
    `NoOverlapObservationRepresentation`, refactored here since DRY

    Args:
        max_type_index (int): highest value the type of the objects can take
        max_state_index (int): highest value the state of the objects can take
        max_color_value (int): highest value colors can take
        width (int): width of the grid
        height (int): height of the grid

    Returns:
        Dict[str, np.ndarray]: {'grid', 'item', 'legacy-agent', 'legacy-grid'}
    """

    rep = default_representation_space(
        max_type_index, max_state_index, max_color_value, width, height
    )

    del rep['agent']

    # increment channels to ensure there is no overlap
    rep['grid'][:, :, 1] += max_type_index + 1
    rep['grid'][:, :, 2] += max_type_index + max_state_index + 2

    # default also returns agent ids as fourth channel, which must be removed
    rep['grid'] = rep['grid'][:, :, :3]

    rep['item'][1] += max_type_index + 1
    rep['item'][2] += max_type_index + max_state_index + 2

    # legacy
    # increment channels to ensure there is no overlap
    rep['legacy-grid'][:, :, [1, 4]] += max_type_index + 1
    rep['legacy-grid'][:, :, [2, 5]] += max_type_index + max_state_index + 2

    # default also returns position and orientation, which must be removed
    rep['legacy-agent'] = rep['legacy-agent'][3:]

    rep['legacy-agent'][1] += max_type_index + 1
    rep['legacy-agent'][2] += max_type_index + max_state_index + 2

    return rep


def no_overlap_convert(
    grid: Grid, agent: Agent, max_type_index: int, max_state_index: int
) -> Dict[str, np.ndarray]:
    """similar to the `default_representation_space`, but ensures no overlap between channels

    Will return a dictionary that contains 'grid' and 'item':

    return['legacy-grid'] is a height x width x 3 shaped array of max values

        Where the values in the first 3 channels have a unique
        meaning. As in, the values in channel 2 (`grid[h,w,2]`) do not overlap
        in any of the other channels. This is to allow specific index numbers
        to have unique meanings. In short, the values in channel 2 start from
        the max value of channel 1, and the values in channel 3 start from the
        max of channel 2 etc.

    return['item'] is a 3-value feature array with max item type/status/color
    channels

        Where the channels reflect those in 'grid'

    NOTE: used by `NoOverlapStateRepresentation` and
    `NoOverlapObservationRepresentation`, refactored here since DRY

    Lastly also returns legacy grid and agent representation

    Args:
        max_type_index (int): highest value the type of the objects can take
        max_state_index (int): highest value the state of the objects can take
        max_color_value (int): highest value colors can take
        width (int): width of the grid
        height (int): height of the grid

    Returns:
        Dict[str, np.ndarray]: {'legacy-grid': array, 'legacy-agent': array}
    """

    rep = default_convert(grid, agent)

    # default also returns agent position, which must be removed
    rep['grid'] = rep['grid'][:, :, :3]

    # increment channels to ensure there is no overlap
    rep['grid'][:, :, 1] += max_type_index + 1
    rep['grid'][:, :, 2] += max_type_index + max_state_index + 2

    rep['item'][1] += max_type_index + 1
    rep['item'][2] += max_type_index + max_state_index + 2

    del rep['agent']

    # legacy
    # default also returns item in the legacy grid, which must be removed
    rep['legacy-agent'] = rep['legacy-agent'][3:]

    # increment channels to ensure there is no overlap
    rep['legacy-grid'][:, :, [1, 4]] += max_type_index + 1
    rep['legacy-grid'][:, :, [2, 5]] += max_type_index + max_state_index + 2

    rep['legacy-agent'][1] += max_type_index + 1
    rep['legacy-agent'][2] += max_type_index + max_state_index + 2

    return rep
