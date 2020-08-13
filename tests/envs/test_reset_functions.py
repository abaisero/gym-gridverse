import random
import unittest

import gym_gridverse.envs.reset_functions as reset_funcs
from gym_gridverse.geometry import Position
from gym_gridverse.grid_object import Door, Goal, Key, Wall


class TestGymDoorKey(unittest.TestCase):
    def setUp(self):
        self.size = random.randint(5, 12)
        self.state = reset_funcs.reset_minigrid_door_key(self.size)

    @property
    def wall_column(self):
        for x in range(1, self.size - 1):
            found_wall = isinstance(self.state.grid[Position(1, x)], Wall)
            found_door = isinstance(self.state.grid[Position(1, x)], Door)
            if found_wall or found_door:
                return x

    def test_throw_if_too_small(self):
        """Asserts method throws if provided size is too small"""

        negative_size = -5
        too_small = 4

        self.assertRaises(
            ValueError, reset_funcs.reset_minigrid_door_key, negative_size
        )
        self.assertRaises(
            ValueError, reset_funcs.reset_minigrid_door_key, too_small
        )

    def test_wall(self):
        """Tests whether the reset state contains a wall column"""

        # Surrounded by walls
        for i in range(0, self.size):
            self.assertIsInstance(self.state.grid[Position(0, i)], Wall)
            self.assertIsInstance(self.state.grid[Position(i, 0)], Wall)

        count = 0
        for x in range(1, self.size - 1):
            found_wall = isinstance(self.state.grid[Position(1, x)], Wall)
            found_door = isinstance(self.state.grid[Position(1, x)], Door)
            if found_wall or found_door:
                count += 1
                column = x
        self.assertEqual(count, 1, "Should only be one column of walls")
        self.assertEqual(self.wall_column, column)

        count = 0
        for y in range(1, self.size - 1):
            if isinstance(self.state.grid[Position(y, column)], Door):
                count += 1
            else:
                self.assertIsInstance(
                    self.state.grid[Position(y, column)], Wall
                )
        self.assertEqual(count, 1, "There should be exactly 1 door")

    def test_agent_is_left_of_wall(self):
        self.assertLess(
            self.state.agent.position.x,
            self.wall_column,
            "Agent should be left of wall",
        )

    def test_key(self):

        count = 0
        for pos in self.state.grid.positions():
            if isinstance(self.state.grid[pos], Key):
                count += 1
                key_pos = pos
        self.assertEqual(count, 1, "There should be only 1 key")

        self.assertLess(
            key_pos.x, self.wall_column, "Key should be left of wall"
        )

    def test_goal(self):
        self.assertIsInstance(
            self.state.grid[Position(self.size - 2, self.size - 2)],
            Goal,
            "There should be a goal left bottom",
        )
