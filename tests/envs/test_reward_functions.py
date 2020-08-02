import unittest

from gym_gridverse.envs.reward_functions import living_reward, reach_goal
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid_object import Goal
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State


def make_goal_state(agent_on_goal: bool) -> State:
    """makes a simple state with goal object and agent on or off the goal"""
    grid = Grid(2, 1)
    grid[Position(0, 0)] = Goal()
    agent_position = Position(0, 0) if agent_on_goal else Position(1, 0)
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


if __name__ == '__main__':
    unittest.main()
