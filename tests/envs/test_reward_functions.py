import unittest

from gym_gridverse.envs.reward_functions import (
    bump_moving_obstacle,
    getting_closer,
    living_reward,
    proportional_to_distance,
    reach_goal,
)
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid_object import Goal, MovingObstacle
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State


def make_5x5_goal_state() -> State:
    """makes a simple 5x5 state with goal object in the middle"""
    grid = Grid(5, 5)
    grid[Position(2, 2)] = Goal()
    agent = Agent(Position(0, 0), Orientation.N)
    return State(grid, agent)


def make_goal_state(agent_on_goal: bool) -> State:
    """makes a simple state with goal object and agent on or off the goal"""
    grid = Grid(2, 1)
    grid[Position(0, 0)] = Goal()
    agent_position = Position(0, 0) if agent_on_goal else Position(1, 0)
    agent = Agent(agent_position, Orientation.N)
    return State(grid, agent)


def make_moving_obstacle_state(agent_on_obstacle: bool) -> State:
    """makes a simple state with goal object and agent on or off the goal"""
    grid = Grid(2, 1)
    grid[Position(0, 0)] = MovingObstacle()
    agent_position = Position(0, 0) if agent_on_obstacle else Position(1, 0)
    agent = Agent(agent_position, Orientation.N)
    return State(grid, agent)


class TestLivingReward(unittest.TestCase):
    def test_living_reward_default(self):
        self.assertEqual(living_reward(None, None, None), -1.0)

    def test_living_reward_custom(self):
        self.assertEqual(living_reward(None, None, None, reward=-1.0), -1.0)
        self.assertEqual(living_reward(None, None, None, reward=0.0), 0.0)
        self.assertEqual(living_reward(None, None, None, reward=1.0), 1.0)


class TestReachGoal(unittest.TestCase):
    def test_reach_goal_default(self):
        # on goal
        next_state = make_goal_state(agent_on_goal=True)
        self.assertEqual(reach_goal(None, None, next_state), 1.0)

        # off goal
        next_state = make_goal_state(agent_on_goal=False)
        self.assertEqual(reach_goal(None, None, next_state), 0.0)

    def test_reach_goal_custom(self):
        # on goal
        next_state = make_goal_state(agent_on_goal=True)
        self.assertEqual(
            reach_goal(
                None, None, next_state, reward_on=10.0, reward_off=-1.0,
            ),
            10.0,
        )

        # off goal
        next_state = make_goal_state(agent_on_goal=False)
        self.assertEqual(
            reach_goal(
                None, None, next_state, reward_on=10.0, reward_off=-1.0,
            ),
            -1.0,
        )


class TestBumpMovingObstacle(unittest.TestCase):
    def test_bump_moving_obstacle_default(self):
        # on goal
        next_state = make_moving_obstacle_state(agent_on_obstacle=True)
        self.assertEqual(bump_moving_obstacle(None, None, next_state), -1.0)

        # off goal
        next_state = make_moving_obstacle_state(agent_on_obstacle=False)
        self.assertEqual(bump_moving_obstacle(None, None, next_state), 0.0)

    def test_bump_moving_obstacle_custom(self):
        # on goal
        next_state = make_moving_obstacle_state(agent_on_obstacle=True)
        self.assertEqual(
            bump_moving_obstacle(None, None, next_state, reward=-10.0), -10.0,
        )

        # off goal
        next_state = make_moving_obstacle_state(agent_on_obstacle=False)
        self.assertEqual(
            bump_moving_obstacle(None, None, next_state, reward=-10.0,), 0.0,
        )


class TestProportionalToDistance(unittest.TestCase):
    def test_proportional_to_distance_default(self):
        state = make_5x5_goal_state()

        # moving agent on the top row

        state.agent.position = Position(0, 0)
        self.assertAlmostEqual(
            proportional_to_distance(None, None, state, object_type=Goal), -4
        )

        state.agent.position = Position(0, 1)
        self.assertAlmostEqual(
            proportional_to_distance(None, None, state, object_type=Goal), -3
        )

        state.agent.position = Position(0, 2)
        self.assertAlmostEqual(
            proportional_to_distance(None, None, state, object_type=Goal), -2
        )

        state.agent.position = Position(0, 3)
        self.assertAlmostEqual(
            proportional_to_distance(None, None, state, object_type=Goal), -3
        )

        state.agent.position = Position(0, 4)
        self.assertAlmostEqual(
            proportional_to_distance(None, None, state, object_type=Goal), -4
        )

        # moving agent on the middle row

        state.agent.position = Position(2, 0)
        self.assertAlmostEqual(
            proportional_to_distance(None, None, state, object_type=Goal), -2
        )

        state.agent.position = Position(2, 1)
        self.assertAlmostEqual(
            proportional_to_distance(None, None, state, object_type=Goal), -1
        )

        state.agent.position = Position(2, 2)
        self.assertAlmostEqual(
            proportional_to_distance(None, None, state, object_type=Goal), 0
        )

        state.agent.position = Position(2, 3)
        self.assertAlmostEqual(
            proportional_to_distance(None, None, state, object_type=Goal), -1
        )

        state.agent.position = Position(2, 4)
        self.assertAlmostEqual(
            proportional_to_distance(None, None, state, object_type=Goal), -2
        )

    def test_proportional_to_distance_custom(self):
        state = make_5x5_goal_state()

        # moving agent on the top row

        state.agent.position = Position(0, 0)
        self.assertAlmostEqual(
            proportional_to_distance(
                None,
                None,
                state,
                object_type=Goal,
                reward_per_unit_distance=0.1,
            ),
            0.4,
        )

        state.agent.position = Position(0, 1)
        self.assertAlmostEqual(
            proportional_to_distance(
                None,
                None,
                state,
                object_type=Goal,
                reward_per_unit_distance=0.1,
            ),
            0.3,
        )

        state.agent.position = Position(0, 2)
        self.assertAlmostEqual(
            proportional_to_distance(
                None,
                None,
                state,
                object_type=Goal,
                reward_per_unit_distance=0.1,
            ),
            0.2,
        )

        state.agent.position = Position(0, 3)
        self.assertAlmostEqual(
            proportional_to_distance(
                None,
                None,
                state,
                object_type=Goal,
                reward_per_unit_distance=0.1,
            ),
            0.3,
        )

        state.agent.position = Position(0, 4)
        self.assertAlmostEqual(
            proportional_to_distance(
                None,
                None,
                state,
                object_type=Goal,
                reward_per_unit_distance=0.1,
            ),
            0.4,
        )

        # moving agent on the middle row

        state.agent.position = Position(2, 0)
        self.assertAlmostEqual(
            proportional_to_distance(
                None,
                None,
                state,
                object_type=Goal,
                reward_per_unit_distance=0.1,
            ),
            0.2,
        )

        state.agent.position = Position(2, 1)
        self.assertAlmostEqual(
            proportional_to_distance(
                None,
                None,
                state,
                object_type=Goal,
                reward_per_unit_distance=0.1,
            ),
            0.1,
        )

        state.agent.position = Position(2, 2)
        self.assertAlmostEqual(
            proportional_to_distance(
                None,
                None,
                state,
                object_type=Goal,
                reward_per_unit_distance=0.1,
            ),
            0.0,
        )

        state.agent.position = Position(2, 3)
        self.assertAlmostEqual(
            proportional_to_distance(
                None,
                None,
                state,
                object_type=Goal,
                reward_per_unit_distance=0.1,
            ),
            0.1,
        )

        state.agent.position = Position(2, 4)
        self.assertAlmostEqual(
            proportional_to_distance(
                None,
                None,
                state,
                object_type=Goal,
                reward_per_unit_distance=0.1,
            ),
            0.2,
        )


class TestGettingCloser(unittest.TestCase):
    def test_getting_closer_default(self):
        state_on_goal = make_goal_state(agent_on_goal=True)
        state_off_goal = make_goal_state(agent_on_goal=False)

        self.assertEqual(
            getting_closer(
                state_off_goal, None, state_off_goal, object_type=Goal
            ),
            0.0,
        )
        self.assertEqual(
            getting_closer(
                state_off_goal, None, state_on_goal, object_type=Goal
            ),
            1.0,
        )
        self.assertEqual(
            getting_closer(
                state_on_goal, None, state_off_goal, object_type=Goal
            ),
            -1.0,
        )
        self.assertEqual(
            getting_closer(
                state_on_goal, None, state_on_goal, object_type=Goal
            ),
            0.0,
        )

    def test_getting_closer_custom(self):
        state_on_goal = make_goal_state(agent_on_goal=True)
        state_off_goal = make_goal_state(agent_on_goal=False)

        self.assertEqual(
            getting_closer(
                state_off_goal,
                None,
                state_off_goal,
                object_type=Goal,
                reward_closer=2.0,
                reward_further=-5.0,
            ),
            0.0,
        )
        self.assertEqual(
            getting_closer(
                state_off_goal,
                None,
                state_on_goal,
                object_type=Goal,
                reward_closer=2.0,
                reward_further=-5.0,
            ),
            2.0,
        )
        self.assertEqual(
            getting_closer(
                state_on_goal,
                None,
                state_off_goal,
                object_type=Goal,
                reward_closer=2.0,
                reward_further=-5.0,
            ),
            -5.0,
        )
        self.assertEqual(
            getting_closer(
                state_on_goal,
                None,
                state_on_goal,
                object_type=Goal,
                reward_closer=2.0,
                reward_further=-5.0,
            ),
            0.0,
        )


if __name__ == '__main__':
    unittest.main()
