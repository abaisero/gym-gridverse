import unittest

from gym_gridverse.envs.terminating_functions import reach_goal
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


class TestReachGoal(unittest.TestCase):
    def test_reach_goal(self):
        # on goal
        next_state = make_goal_state(agent_on_goal=True)
        self.assertTrue(reach_goal(None, None, next_state))

        # off goal
        next_state = make_goal_state(agent_on_goal=False)
        self.assertFalse(reach_goal(None, None, next_state))


if __name__ == '__main__':
    unittest.main()
