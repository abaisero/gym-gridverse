import abc
from typing import Dict

import numpy as np

from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import GridObject, NoneGridObject
from gym_gridverse.info import Agent, Grid
from gym_gridverse.observation import Observation
from gym_gridverse.state import State


class Representation(metaclass=abc.ABCMeta):
    """Base interface for state, observation and object representation"""

    @property
    @abc.abstractmethod
    def space(self) -> Dict[str, np.ndarray]:
        """range of values the representation can return

        returns a string -> numpy array of max values, e.g:
        same shape as `convert` but returns max values
        """


class StateRepresentation(Representation):
    """Base interface for state representations: enforces `convert`"""

    @abc.abstractmethod
    def convert(self, s: State) -> Dict[str, np.ndarray]:
        """returns state `s` representation as str -> array dict"""


class ObservationRepresentation(Representation):
    """Base interface for observation representations: enforces `convert`"""

    @abc.abstractmethod
    def convert(self, o: Observation) -> Dict[str, np.ndarray]:
        """returns observation `o` representation as str -> array dict"""


class GridObjectRepresentation(Representation):
    """Base interface for observation representations: enforces `convert`

    XXX: currently not in use in the code base, at some point we may want to
    implement this because we will like it """

    @abc.abstractmethod
    def convert(self, obj: GridObject) -> Dict[str, np.ndarray]:
        """returns grid object `obj` representation as str -> array dict"""


def default_representation_space(
    max_type_index: int,
    max_state_index: int,
    max_color_value: int,
    width: int,
    height: int,
) -> Dict[str, np.ndarray]:
    """the naive space of the representation, returns max indices each value can take

    NOTE: used by `DefaultStateRepresentation` and
    `DefaultObservationRepresentation`, refactored here since DRY

    return['grid'] is a height x width x 5 shapes array of max values
    return['agent'] is a 6-value feature array representing the agent
    pos/orientation and held object

    Args:
        max_type_index (int): highest value the type of the objects can take
        max_state_index (int): highest value the state of the objects can take
        max_color_value (int): highest value colors can take
        width (int): width of the grid
        height (int): height of the grid

    Returns:
        Dict[str, np.ndarray]: {'grid': array, 'agent': array}
    """
    assert min([max_type_index, width, height]) > 0, str(
        [max_type_index, width, height]
    )

    grid_array = np.array(
        [[[max_type_index, max_state_index, max_color_value] * 2] * width]
        * height
    )
    agent_array = np.array(
        [
            height,
            width,
            len(Orientation),
            max_type_index,
            max_state_index,
            max_color_value,
        ]
    )
    return {'grid': grid_array, 'agent': agent_array}


def default_convert(grid: Grid, agent: Agent) -> Dict[str, np.ndarray]:
    """Default naive convertion of a grid and agent

    Converts grid to a 6 channel (of height x width) representation of:
        0. object type index
        1. object state index
        2. object color index
        3. zeros except for agent's position which is agent object's type index
        4. zeros except for agent's position which is agent object's state index
        5. zeros except for agent's position which is agent object's color index

    e.g. return['grid'][h,w,2] returns the color of object on grid position (h,w)

    Converts agent into a 3 feature value: [
        agent y position,
        agent x positoin,
        agent orientation,
        item type index,
        item status index,
        item color index
    ]

    NOTE: used by `DefaultStateRepresentation` and
    `DefaultObservationRepresentation`, refactored here since DRY

    Args:
        grid (Grid):
        agent (Agent):

    Returns:
        Dict[str, np.ndarray]: {'grid': array, 'state': array}
    """

    agent_obj_array = np.array(
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
    none_grid_object = NoneGridObject()
    grid_array_agent_channels = np.array(
        [
            [
                [
                    none_grid_object.type_index,  # pylint: disable=no-member
                    none_grid_object.state_index,
                    none_grid_object.color.value,
                ]
                for x in range(grid.shape.width)
            ]
            for y in range(grid.shape.height)
        ]
    )
    grid_array_agent_channels[agent.position] = agent_obj_array

    grid_array = np.concatenate(
        (grid_array_agent_channels, grid_array_object_channels), axis=-1
    )

    agent_array = np.concatenate(
        [
            [agent.position.y, agent.position.x, agent.orientation.value],
            agent_obj_array,
        ],
    )

    return {'grid': grid_array, 'agent': agent_array}


def no_overlap_representation_space(
    max_type_index: int,
    max_state_index: int,
    max_color_value: int,
    width: int,
    height: int,
) -> Dict[str, np.ndarray]:
    """similar to the `default_representation_space`, but ensures no overlap between channels

    Will return a dictionary that contains 'grid' and 'agent':

    return['grid'] is a height x width x 5 shaped array of max values

        However, where the values in the first 3 channels have a unique
        meaning. As in, the values in channel 2 (`grid[h,w,2]`) do not overlap
        in any of the other channels. This is to allow specific index numbers
        to have unique meanings. In short, the values in channel 2 start from
        the max value of channel 1, and the values in channel 3 start from the
        max of channel 2 etc.

    return['agent'] is a 3-value feature array with max item type/status/color
    channels

        Where the channels reflect those in 'grid'

    NOTE: used by `NoOverlapStateRepresentation` and
    `NoOverlapObservationRepresentation`, refactored here since DRY

    Args:
        max_type_index (int): highest value the type of the objects can take
        max_state_index (int): highest value the state of the objects can take
        max_color_value (int): highest value colors can take
        width (int): width of the grid
        height (int): height of the grid

    Returns:
        Dict[str, np.ndarray]: {'grid': array, 'agent': array}
    """

    rep = default_representation_space(
        max_type_index, max_state_index, max_color_value, width, height
    )

    # increment channels to ensure there is no overlap
    rep['grid'][:, :, [1, 4]] += max_type_index
    rep['grid'][:, :, [2, 5]] += max_type_index + max_state_index

    # default also returns position and orientation, which must be removed
    rep['agent'] = rep['agent'][3:]

    rep['agent'][1] += max_type_index
    rep['agent'][2] += max_type_index + max_state_index

    return rep


def no_overlap_convert(
    grid: Grid, agent: Agent, max_type_index: int, max_state_index: int,
) -> Dict[str, np.ndarray]:
    """similar to the `default_representation_space`, but ensures no overlap between channels

    Will return a dictionary that contains 'grid' and 'agent':

    return['grid'] is a height x width x 5 shaped array of max values

        However, where the values in the first 3 channels have a unique
        meaning. As in, the values in channel 2 (`grid[h,w,2]`) do not overlap
        in any of the other channels. This is to allow specific index numbers
        to have unique meanings. In short, the values in channel 2 start from
        the max value of channel 1, and the values in channel 3 start from the
        max of channel 2 etc.

    return['agent'] is a 3-value feature array with max item type/status/color
    channels

        Where the channels reflect those in 'grid'

    NOTE: used by `NoOverlapStateRepresentation` and
    `NoOverlapObservationRepresentation`, refactored here since DRY

    Args:
        max_type_index (int): highest value the type of the objects can take
        max_state_index (int): highest value the state of the objects can take
        max_color_value (int): highest value colors can take
        width (int): width of the grid
        height (int): height of the grid

    Returns:
        Dict[str, np.ndarray]: {'grid': array, 'agent': array}
    """

    rep = default_convert(grid, agent)

    # increment channels to ensure there is no overlap
    rep['grid'][:, :, [1, 4]] += max_type_index
    rep['grid'][:, :, [2, 5]] += max_type_index + max_state_index

    # default also returns position and orientation, which must be removed
    rep['agent'] = rep['agent'][3:]

    rep['agent'][1] += max_type_index
    rep['agent'][2] += max_type_index + max_state_index

    return rep
