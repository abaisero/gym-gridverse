import unittest

from gym_gridverse.actions import Actions
from gym_gridverse.envs.terminating_functions import bump_into_wall, reach_goal
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid_object import Goal, Wall
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


def make_wall_state() -> State:
    """makes a simple state with Wall object and agent in front of it"""
    grid = Grid(2, 1)
    grid[Position(0, 0)] = Wall()
    agent_position = Position(1, 0)
    agent = Agent(agent_position, Orientation.N)
    return State(grid, agent)


class TestBumpIntoWall(unittest.TestCase):
    def test_valid_moves(self):
        state = make_wall_state()

        self.assertFalse(bump_into_wall(state, Actions.MOVE_LEFT, None))
        self.assertFalse(bump_into_wall(state, Actions.TURN_RIGHT, None))
        self.assertFalse(bump_into_wall(state, Actions.ACTUATE, None))

    def test_bumps(self):

        state = make_wall_state()
        self.assertTrue(bump_into_wall(state, Actions.MOVE_FORWARD, None))

        state.agent.orientation = Orientation.W
        self.assertTrue(bump_into_wall(state, Actions.MOVE_RIGHT, None))


if __name__ == '__main__':
    unittest.main()
