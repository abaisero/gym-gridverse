""" Tests state dynamics """

import copy
import random
import unittest

from gym_gridverse.actions import Actions
from gym_gridverse.envs.state_dynamics import (
    move_agent,
    pickup_mechanics,
    rotate_agent,
)
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Floor,
    Key,
    NoneGridObject,
    Wall,
)
from gym_gridverse.info import Agent, Grid, Orientation, Position
from gym_gridverse.state import State


class TestRotateAgent(unittest.TestCase):
    @staticmethod
    def random_object():
        random.choice([Key, Floor])

    def test_the_rotation(self):
        random_position = Position(random.randint(0, 10), random.randint(0, 10))
        random_object = self.random_object()

        # Facing NORTH
        agent = Agent(random_position, Orientation.N, random_object)

        # Rotate to WEST
        rotate_agent(agent, Actions.TURN_LEFT)
        self.assertEqual(agent.orientation, Orientation.W)

        # Not rotating
        rotate_agent(agent, Actions.MOVE_LEFT)
        rotate_agent(agent, Actions.ACTUATE)
        rotate_agent(agent, Actions.PICK_N_DROP)
        self.assertEqual(agent.orientation, Orientation.W)

        # Two rotation to EAST
        rotate_agent(agent, Actions.TURN_LEFT)
        rotate_agent(agent, Actions.TURN_LEFT)
        self.assertEqual(agent.orientation, Orientation.E)

        # One back to SOUTH
        rotate_agent(agent, Actions.TURN_RIGHT)
        self.assertEqual(agent.orientation, Orientation.S)

        # Full circle for fun to SOUTH
        rotate_agent(agent, Actions.TURN_RIGHT)
        rotate_agent(agent, Actions.TURN_RIGHT)
        rotate_agent(agent, Actions.TURN_RIGHT)
        rotate_agent(agent, Actions.TURN_RIGHT)
        self.assertEqual(agent.orientation, Orientation.S)


class TestMoveAgent(unittest.TestCase):
    def setUp(self):
        """Sets up a 3x2 Grid with agent facing north in (2,1)"""
        self.grid = Grid(height=3, width=2)
        self.agent = Agent(position=Position(2, 1), orientation=Orientation.N)

    def test_basic_movement_north(self):
        """Test unblocked movement"""

        # Move to (2,0)
        move_agent(self.agent, self.grid, action=Actions.MOVE_LEFT)
        self.assertEqual(self.agent.position, Position(2, 0))

        # Move to (2,1)
        move_agent(self.agent, self.grid, action=Actions.MOVE_RIGHT)
        self.assertEqual(self.agent.position, Position(2, 1))

        # Move up twice to (0,1)
        move_agent(self.agent, self.grid, action=Actions.MOVE_FORWARD)
        move_agent(self.agent, self.grid, action=Actions.MOVE_FORWARD)
        self.assertEqual(self.agent.position, Position(0, 1))

        # Move down once to (1,1)
        move_agent(self.agent, self.grid, action=Actions.MOVE_BACKWARD)
        self.assertEqual(self.agent.position, Position(1, 1))

    def test_blocked_by_grid_object(self):
        """ Puts an object on (2,0) and try to move there"""
        self.grid[Position(2, 0)] = Door(Door.Status.CLOSED, Colors.YELLOW)
        move_agent(self.agent, self.grid, action=Actions.MOVE_LEFT)

        self.assertEqual(self.agent.position, Position(2, 1))

    def test_blocked_by_edges(self):
        """Verify agent does not move outside of bounds"""
        move_agent(self.agent, self.grid, action=Actions.MOVE_RIGHT)

        self.assertEqual(self.agent.position, Position(2, 1))

    def test_other_actions_are_inactive(self):
        """Make sure actions that do not _move_ are ignored"""
        move_agent(self.agent, self.grid, action=Actions.TURN_RIGHT)
        move_agent(self.agent, self.grid, action=Actions.PICK_N_DROP)
        move_agent(self.agent, self.grid, action=Actions.ACTUATE)

        self.assertEqual(self.agent.position, Position(2, 1))

    def test_facing_east(self):
        """Just another test to make sure orientation works as intended"""
        self.agent.orientation = Orientation.E

        move_agent(self.agent, self.grid, action=Actions.MOVE_LEFT)
        move_agent(self.agent, self.grid, action=Actions.MOVE_LEFT)
        self.assertEqual(self.agent.position, Position(0, 1))

        move_agent(self.agent, self.grid, action=Actions.MOVE_BACKWARD)
        self.assertEqual(self.agent.position, Position(0, 0))

        move_agent(self.agent, self.grid, action=Actions.MOVE_FORWARD)
        self.assertEqual(self.agent.position, Position(0, 1))
        move_agent(self.agent, self.grid, action=Actions.MOVE_FORWARD)
        self.assertEqual(self.agent.position, Position(0, 1))

        move_agent(self.agent, self.grid, action=Actions.MOVE_RIGHT)
        self.assertEqual(self.agent.position, Position(1, 1))

    def test_can_go_on_non_block_objects(self):
        self.grid[Position(2, 0)] = Door(Door.Status.OPEN, Colors.YELLOW)
        move_agent(self.agent, self.grid, action=Actions.MOVE_LEFT)

        self.assertEqual(self.agent.position, Position(2, 0))

        self.grid[Position(2, 1)] = Key(Colors.BLUE)
        move_agent(self.agent, self.grid, action=Actions.MOVE_RIGHT)

        self.assertEqual(self.agent.position, Position(2, 1))


class TestPickupMechanics(unittest.TestCase):
    def setUp(self):
        """Sets up a 3x4 Grid with agent facing south in (1,2)"""
        self.grid = Grid(height=3, width=4)
        self.agent = Agent(position=Position(1, 2), orientation=Orientation.S)
        self.a = Actions.PICK_N_DROP
        self.item_pos = Position(2, 2)

    @staticmethod
    def step_with_copy(s: State, a: Actions):
        next_s = copy.deepcopy(s)
        pickup_mechanics(next_s, a)

        return next_s

    def test_nothing_to_pickup(self):
        s = State(self.grid, self.agent)

        # Cannot pickup floor
        next_s = self.step_with_copy(s, self.a)
        self.assertEqual(s, next_s)

        # Cannot pickup door
        self.grid[self.item_pos] = Door(Door.Status.CLOSED, Colors.GREEN)
        next_s = self.step_with_copy(s, self.a)
        self.assertEqual(s, next_s)

        self.assertTrue(isinstance(next_s.grid[self.item_pos], Door))

    def test_pickup(self):
        self.grid[self.item_pos] = Key(Colors.GREEN)
        s = State(self.grid, self.agent)

        # Pick up works
        next_s = self.step_with_copy(s, self.a)
        self.assertEqual(self.grid[self.item_pos], next_s.agent.obj)
        self.assertIsInstance(next_s.grid[self.item_pos], Floor)

        # Pick up only works with correct action
        next_s = self.step_with_copy(s, Actions.MOVE_LEFT)
        self.assertNotEqual(self.grid[self.item_pos], next_s.agent.obj)
        self.assertEqual(next_s.grid[self.item_pos], self.grid[self.item_pos])

    def test_drop(self):
        self.agent.obj = Key(Colors.BLUE)
        s = State(self.grid, self.agent)

        # Can drop:
        next_s = self.step_with_copy(s, self.a)
        self.assertIsInstance(next_s.agent.obj, NoneGridObject)
        self.assertEqual(self.agent.obj, next_s.grid[self.item_pos])

        # Cannot drop:
        s.grid[self.item_pos] = Wall()

        next_s = self.step_with_copy(s, self.a)
        self.assertIsInstance(next_s.grid[self.item_pos], Wall)
        self.assertEqual(self.agent.obj, next_s.agent.obj)

    def test_swap(self):
        self.agent.obj = Key(Colors.BLUE)
        self.grid[self.item_pos] = Key(Colors.GREEN)
        s = State(self.grid, self.agent)

        next_s = self.step_with_copy(s, self.a)
        self.assertEqual(s.grid[self.item_pos], next_s.agent.obj)
        self.assertEqual(s.agent.obj, next_s.grid[self.item_pos])
