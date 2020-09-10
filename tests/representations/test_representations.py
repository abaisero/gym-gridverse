import unittest

import numpy as np

from gym_gridverse.envs.reset_functions import reset_minigrid_empty
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Floor,
    Goal,
    Key,
    NoneGridObject,
    Wall,
)
from gym_gridverse.info import Agent, Grid
from gym_gridverse.representations.representation import (
    default_convert,
    default_representation_space,
    no_overlap_convert,
    no_overlap_representation_space,
)


class TestDefaultRepresentation(unittest.TestCase):
    def setUp(self):
        """Creates an observation representation of 3 by 3"""
        self.h, self.w = 3, 3

        self.max_obj_type = Key.type_index  # pylint: disable=no-member
        self.max_obj_state = Door.num_states()
        self.max_color_value = Colors.BLUE.value

    def test_space(self):  # pylint: disable=no-self-use

        max_channel_values = [
            self.max_obj_type,
            self.max_obj_state,
            self.max_color_value,
        ]
        expected_grid_space = np.array(
            [[max_channel_values * 2] * self.w] * self.h
        )

        space = default_representation_space(
            self.max_obj_type,
            self.max_obj_state,
            self.max_color_value,
            self.w,
            self.h,
        )

        np.testing.assert_array_equal(space['grid'], expected_grid_space)
        np.testing.assert_array_equal(space['agent'], max_channel_values)

    def test_convert(self):

        agent = Agent(Position(0, 2), Orientation.N, Key(Colors.RED))
        grid = Grid(self.h, self.w)
        grid[1, 1] = Door(Door.Status.CLOSED, Colors.BLUE)

        floor_index = Floor.type_index  # pylint: disable=no-member

        expected_agent_observation = np.array(
            [
                agent.obj.type_index,
                agent.obj.state_index,
                agent.obj.color.value,
            ]
        )

        expected_agent_position_channels = np.zeros((3, 3, 3))
        # pylint: disable=no-member
        expected_agent_position_channels[:, :, 0] = NoneGridObject.type_index
        expected_agent_position_channels[:, :, 1] = 0  # status
        expected_agent_position_channels[:, :, 2] = Colors.NONE.value

        expected_agent_position_channels[0, 2] = expected_agent_observation

        expected_grid_observation = np.array(
            [
                [
                    [floor_index, 0, 0],
                    [floor_index, 0, 0],
                    [floor_index, 0, 0],
                ],
                [
                    [floor_index, 0, 0],
                    [
                        Door.type_index,  # pylint: disable=no-member
                        Door.Status.CLOSED.value,
                        Colors.BLUE.value,
                    ],
                    [floor_index, 0, 0],
                ],
                [
                    [floor_index, 0, 0],
                    [floor_index, 0, 0],
                    [floor_index, 0, 0],
                ],
            ]
        )

        rep = default_convert(grid, agent)

        np.testing.assert_array_equal(
            rep['grid'][:, :, :3], expected_agent_position_channels
        )
        np.testing.assert_array_equal(
            rep['grid'][:, :, 3:], expected_grid_observation
        )
        np.testing.assert_array_equal(rep['agent'], expected_agent_observation)


class TestNoOverlapRepresentation(unittest.TestCase):
    def setUp(self):
        """Creates an state representation of 4 by 5"""
        self.h, self.w = 4, 5

        # hard coded from above
        self.max_object_type = Goal.type_index  # pylint: disable=no-member
        self.max_object_status = 0
        self.max_color_value = Colors.GREEN.value

    def test_space(self):  # pylint: disable=no-self-use

        max_channel_values = [
            self.max_object_type,
            self.max_object_type + self.max_object_status,
            self.max_object_type
            + self.max_object_status
            + self.max_color_value,
        ]

        expected_grid_space = np.array(
            [[max_channel_values * 2] * self.w] * self.h
        )

        space = no_overlap_representation_space(
            self.max_object_type,
            self.max_object_status,
            self.max_color_value,
            self.w,
            self.h,
        )

        np.testing.assert_array_equal(space['grid'], expected_grid_space)
        np.testing.assert_array_equal(space['agent'], max_channel_values)

    def test_convert(self):
        state = reset_minigrid_empty(self.h, self.w, random_agent=True)

        # pylint: disable=no-member
        expected_agent_state = np.array(
            [
                NoneGridObject.type_index,
                self.max_object_type,
                self.max_object_type + self.max_object_status,
            ]
        )

        expected_agent_position_channels = np.zeros((self.h, self.w, 3))
        expected_agent_position_channels[:, :] = [
            NoneGridObject.type_index,  # pylint: disable=no-member
            self.max_object_type,  # status
            self.max_object_status + self.max_object_type + Colors.NONE.value,
        ]

        expected_agent_position_channels[
            state.agent.position[0], state.agent.position[1]
        ] = expected_agent_state

        expected_grid_state = np.array(
            [
                [
                    [
                        Floor.type_index,
                        self.max_object_type,
                        self.max_object_type + self.max_object_status,
                    ]
                ]
                * self.w
            ]
            * self.h
        )

        # we expect walls to be around
        expected_grid_state[0, :] = [
            Wall.type_index,
            self.max_object_type,
            self.max_object_type + self.max_object_status,
        ]
        expected_grid_state[self.h - 1, :] = [
            Wall.type_index,
            self.max_object_type,
            self.max_object_type + self.max_object_status,
        ]
        expected_grid_state[:, 0] = [
            Wall.type_index,
            self.max_object_type,
            self.max_object_type + self.max_object_status,
        ]
        expected_grid_state[:, self.w - 1] = [
            Wall.type_index,
            self.max_object_type,
            self.max_object_type + self.max_object_status,
        ]

        # we expect goal to be in corner
        expected_grid_state[self.h - 2, self.w - 2, :] = [
            Goal.type_index,
            self.max_object_type,
            self.max_object_type + self.max_object_status,
        ]

        state_as_array = no_overlap_convert(
            state.grid,
            state.agent,
            self.max_object_type,
            self.max_object_status,
        )

        np.testing.assert_array_equal(
            state_as_array['grid'][:, :, :3], expected_agent_position_channels
        )
        np.testing.assert_array_equal(
            state_as_array['grid'][:, :, 3:], expected_grid_state
        )
        np.testing.assert_array_equal(
            state_as_array['agent'], expected_agent_state
        )
