import unittest
from unittest.mock import Mock

from gym_gridverse.envs.observation_functions import (
    factory,
    minigrid_observation,
)
from gym_gridverse.geometry import Orientation, Position, Shape
from gym_gridverse.grid_object import Floor, Hidden, Wall
from gym_gridverse.info import Agent, Grid
from gym_gridverse.spaces import ObservationSpace
from gym_gridverse.state import State


class Test_MinigridObservation(unittest.TestCase):
    def test_observation(self):
        grid = Grid(10, 10)
        grid[Position(5, 5)] = Wall()

        agent = Agent(Position(7, 7), Orientation.N)
        state = State(grid, agent)
        observation_space = ObservationSpace(Shape(6, 5), [], [])
        observation = minigrid_observation(
            state, observation_space=observation_space
        )
        self.assertEqual(observation.agent.position, Position(5, 2))
        self.assertEqual(observation.agent.orientation, Orientation.N)
        self.assertEqual(observation.agent.obj, state.agent.obj)
        self.assertTupleEqual(observation.grid.shape, (6, 5))
        self.assertIsInstance(observation.grid[Position(3, 0)], Wall)

        agent = Agent(Position(3, 3), Orientation.S)
        state = State(grid, agent)
        observation_space = ObservationSpace(Shape(6, 5), [], [])
        observation = minigrid_observation(
            state, observation_space=observation_space
        )
        self.assertEqual(observation.agent.position, Position(5, 2))
        self.assertEqual(observation.agent.orientation, Orientation.N)
        self.assertEqual(observation.agent.obj, state.agent.obj)
        self.assertTupleEqual(observation.grid.shape, (6, 5))
        self.assertIsInstance(observation.grid[Position(3, 0)], Wall)

        agent = Agent(Position(7, 3), Orientation.E)
        state = State(grid, agent)
        observation_space = ObservationSpace(Shape(6, 5), [], [])
        observation = minigrid_observation(
            state, observation_space=observation_space
        )
        self.assertEqual(observation.agent.position, Position(5, 2))
        self.assertEqual(observation.agent.orientation, Orientation.N)
        self.assertEqual(observation.agent.obj, state.agent.obj)
        self.assertTupleEqual(observation.grid.shape, (6, 5))
        self.assertIsInstance(observation.grid[Position(3, 0)], Wall)

        agent = Agent(Position(3, 7), Orientation.W)
        state = State(grid, agent)
        observation_space = ObservationSpace(Shape(6, 5), [], [])
        observation = minigrid_observation(
            state, observation_space=observation_space
        )
        self.assertEqual(observation.agent.position, Position(5, 2))
        self.assertEqual(observation.agent.orientation, Orientation.N)
        self.assertEqual(observation.agent.obj, state.agent.obj)
        self.assertTupleEqual(observation.grid.shape, (6, 5))
        self.assertIsInstance(observation.grid[Position(3, 0)], Wall)

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
        observation_space = ObservationSpace(Shape(6, 5), [], [])
        observation = minigrid_observation(
            state, observation_space=observation_space
        )
        observation_grid_expected = Grid.from_objects(
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Wall(), Wall(), Wall(), Hidden(),],
                [Hidden(), Floor(), Floor(), Floor(), Hidden(),],
            ]
        )
        self.assertEqual(observation.grid, observation_grid_expected)

        agent = Agent(Position(0, 1), Orientation.S)
        state = State(grid, agent)
        observation_space = ObservationSpace(Shape(6, 5), [], [])
        observation = minigrid_observation(
            state, observation_space=observation_space
        )
        observation_grid_expected = Grid.from_objects(
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Wall(), Wall(), Wall(), Hidden(),],
                [Hidden(), Floor(), Floor(), Floor(), Hidden(),],
            ]
        )
        self.assertEqual(observation.grid, observation_grid_expected)

        agent = Agent(Position(2, 1), Orientation.E)
        state = State(grid, agent)
        observation_space = ObservationSpace(Shape(6, 5), [], [])
        observation = minigrid_observation(
            state, observation_space=observation_space
        )
        observation_grid_expected = Grid.from_objects(
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Wall(), Floor(), Hidden(), Hidden(),],
                [Hidden(), Wall(), Floor(), Hidden(), Hidden(),],
            ]
        )
        self.assertEqual(observation.grid, observation_grid_expected)

        agent = Agent(Position(2, 1), Orientation.W)
        state = State(grid, agent)
        observation_space = ObservationSpace(Shape(6, 5), [], [])
        observation = minigrid_observation(
            state, observation_space=observation_space
        )
        observation_grid_expected = Grid.from_objects(
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Floor(), Wall(), Hidden(),],
                [Hidden(), Hidden(), Floor(), Wall(), Hidden(),],
            ]
        )
        self.assertEqual(observation.grid, observation_grid_expected)


class TestFactory(unittest.TestCase):
    def test_invalid(self):
        self.assertRaises(ValueError, factory, 'invalid')

    def test_valid(self):
        observation_space = Mock()
        factory('full_visibility', observation_space=observation_space)
        self.assertRaises(ValueError, factory, 'full_visibility')

        observation_space = Mock()
        visibility_function = Mock()
        factory(
            'from_visibility',
            observation_space=observation_space,
            visibility_function=visibility_function,
        )
        self.assertRaises(ValueError, factory, 'from_visibility')

        observation_space = Mock()
        factory('minigrid_observation', observation_space=observation_space)
        self.assertRaises(ValueError, factory, 'minigrid_observation')

        observation_space = Mock()
        factory('raytracing_observation', observation_space=observation_space)
        self.assertRaises(ValueError, factory, 'raytracing_observation')

        observation_space = Mock()
        factory(
            'stochastic_raytracing_observation',
            observation_space=observation_space,
        )
        self.assertRaises(
            ValueError, factory, 'stochastic_raytracing_observation'
        )


if __name__ == '__main__':
    unittest.main()
