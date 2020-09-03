import unittest

import numpy as np
from gym_gridverse.geometry import Orientation, Position, Shape
from gym_gridverse.grid_object import Colors, Door, Floor, Key
from gym_gridverse.info import Agent, Grid
from gym_gridverse.observation import Observation
from gym_gridverse.representations.observation_representations import \
    DefaultObservationRepresentation
from gym_gridverse.spaces import ObservationSpace


class TestDefaultObservationRepresentation(unittest.TestCase):
    def setUp(self):
        objects = [Floor, Key, Door]
        colors = [Colors.NONE, Colors.RED, Colors.BLUE]
        self.obs_space = ObservationSpace(Shape(3, 3), objects, colors)

        self.obs_rep = DefaultObservationRepresentation(self.obs_space)

    def test_space(self):  # pylint: disable=no-self-use

        max_channel_values = [
            Key.type_index,  # pylint: disable=no-member
            Door.num_states(),
            Colors.BLUE.value,
        ]
        expected_grid_space = np.array(
            [[max_channel_values] * self.obs_space.grid_shape.width]
            * self.obs_space.grid_shape.height
        )

        space = self.obs_rep.space

        np.testing.assert_array_equal(space['grid'], expected_grid_space)
        np.testing.assert_array_equal(space['agent'], max_channel_values)

    def test_convert(self):
        agent = Agent(Position(0, 0), Orientation.N, Key(Colors.RED))
        grid = Grid(*self.obs_space.grid_shape)
        grid[1, 1] = Door(Door.Status.CLOSED, Colors.BLUE)

        observation = Observation(grid, agent)

        floor_index = Floor.type_index  # pylint: disable=no-member
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

        expected_agent_observation = np.array(
            [
                agent.obj.type_index,
                agent.obj.state_index,
                agent.obj.color.value,
            ]
        )

        obs = self.obs_rep.convert(observation)
        np.testing.assert_array_equal(
            obs['grid'][:, :, 3:], expected_grid_observation
        )
        np.testing.assert_array_equal(obs['agent'], expected_agent_observation)
