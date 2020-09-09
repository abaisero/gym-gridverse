import unittest

import numpy as np
from gym_gridverse.envs.reset_functions import reset_minigrid_empty
from gym_gridverse.geometry import Shape
from gym_gridverse.grid_object import Colors, Floor, Goal, NoneGridObject, Wall
from gym_gridverse.representations.state_representations import \
    DefaultStateRepresentation
from gym_gridverse.spaces import StateSpace


class TestDefaultStateRepresentation(unittest.TestCase):
    def setUp(self):
        """Creates an state representation of 4 by 5"""
        self.h, self.w = 4, 5
        self.objects = [Floor, Wall, Goal]
        self.colors = [Colors.NONE, Colors.GREEN]

        self.state_space = StateSpace(
            Shape(self.h, self.w), self.objects, self.colors
        )
        self.state_rep = DefaultStateRepresentation(self.state_space)

    def test_space(self):  # pylint: disable=no-self-use

        max_channel_values = [
            Goal.type_index,  # pylint: disable=no-member
            Goal.num_states(),
            Colors.GREEN.value,
        ]

        expected_grid_space = np.array([[max_channel_values] * self.w] * self.h)

        space = self.state_rep.space

        np.testing.assert_array_equal(space['grid'], expected_grid_space)
        np.testing.assert_array_equal(space['agent'], max_channel_values)

    def test_convert(self):
        state = reset_minigrid_empty(self.h, self.w, random_agent=True)

        # pylint: disable=no-member
        expected_agent_state = np.array([NoneGridObject.type_index, 0, 0])

        expected_agent_position_channels = np.ones((self.h, self.w, 3))
        expected_agent_position_channels[:, :, 0] = NoneGridObject.type_index
        expected_agent_position_channels[:, :, 1] = 0  # status
        expected_agent_position_channels[:, :, 2] = Colors.NONE.value

        expected_agent_position_channels[
            state.agent.position[0], state.agent.position[1]
        ] = expected_agent_state

        expected_grid_state = np.array(
            [[[Floor.type_index, 0, 0]] * self.w] * self.h
        )

        # we expect walls to be around
        expected_grid_state[0, :] = [Wall.type_index, 0, 0]
        expected_grid_state[self.h - 1, :] = [Wall.type_index, 0, 0]
        expected_grid_state[:, 0] = [Wall.type_index, 0, 0]
        expected_grid_state[:, self.w - 1] = [Wall.type_index, 0, 0]

        # we expect goal to be in corner
        expected_grid_state[self.h - 2, self.w - 2, :] = [Goal.type_index, 0, 0]

        state_as_array = self.state_rep.convert(state)

        np.testing.assert_array_equal(
            state_as_array['grid'][:, :, :3], expected_agent_position_channels
        )
        np.testing.assert_array_equal(
            state_as_array['grid'][:, :, 3:], expected_grid_state
        )
        np.testing.assert_array_equal(
            state_as_array['agent'], expected_agent_state
        )
