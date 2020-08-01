import unittest

from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid_object import Floor, Hidden, Wall
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State


class TestState(unittest.TestCase):
    def test_observation(self):
        grid = Grid(10, 10)
        grid[Position(5, 5)] = Wall()

        agent = Agent(Position(7, 7), Orientation.N)
        state = State(grid, agent)
        observation = state.observation()
        self.assertIs(observation.agent, state.agent)
        self.assertTupleEqual(observation.grid.shape, (7, 7))
        self.assertIsInstance(observation.grid[Position(4, 1)], Wall)

        agent = Agent(Position(3, 3), Orientation.S)
        state = State(grid, agent)
        observation = state.observation()
        self.assertIs(observation.agent, state.agent)
        self.assertTupleEqual(observation.grid.shape, (7, 7))
        self.assertIsInstance(observation.grid[Position(4, 1)], Wall)

        agent = Agent(Position(7, 3), Orientation.E)
        state = State(grid, agent)
        observation = state.observation()
        self.assertIs(observation.agent, state.agent)
        self.assertTupleEqual(observation.grid.shape, (7, 7))
        self.assertIsInstance(observation.grid[Position(4, 1)], Wall)

        agent = Agent(Position(3, 7), Orientation.W)
        state = State(grid, agent)
        observation = state.observation()
        self.assertIs(observation.agent, state.agent)
        self.assertTupleEqual(observation.grid.shape, (7, 7))
        self.assertIsInstance(observation.grid[Position(4, 1)], Wall)

    def test_observation_partially_observable(self):
        grid = Grid.from_objects(
            [
                [Floor(), Floor(), Floor()],
                [Wall(), Wall(), Wall()],
                [Floor(), Floor(), Floor()],
            ]
        )

        agent = Agent(Position(2, 1), Orientation.N)
        state = State(grid, agent)
        observation = state.observation()
        observation_grid_expected = Grid.from_objects(
            [
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Wall(),
                    Wall(),
                    Wall(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Floor(),
                    Floor(),
                    Floor(),
                    Hidden(),
                    Hidden(),
                ],
            ]
        )
        self.assertEqual(observation.grid, observation_grid_expected)

        agent = Agent(Position(0, 1), Orientation.S)
        state = State(grid, agent)
        observation = state.observation()
        observation_grid_expected = Grid.from_objects(
            [
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Wall(),
                    Wall(),
                    Wall(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Floor(),
                    Floor(),
                    Floor(),
                    Hidden(),
                    Hidden(),
                ],
            ]
        )
        self.assertEqual(observation.grid, observation_grid_expected)

        agent = Agent(Position(2, 1), Orientation.E)
        state = State(grid, agent)
        observation = state.observation()
        observation_grid_expected = Grid.from_objects(
            [
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Wall(),
                    Floor(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Wall(),
                    Floor(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
            ]
        )
        self.assertEqual(observation.grid, observation_grid_expected)

        agent = Agent(Position(2, 1), Orientation.W)
        state = State(grid, agent)
        observation = state.observation()
        observation_grid_expected = Grid.from_objects(
            [
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Floor(),
                    Wall(),
                    Hidden(),
                    Hidden(),
                ],
                [
                    Hidden(),
                    Hidden(),
                    Hidden(),
                    Floor(),
                    Wall(),
                    Hidden(),
                    Hidden(),
                ],
            ]
        )
        self.assertEqual(observation.grid, observation_grid_expected)


if __name__ == '__main__':
    unittest.main()
