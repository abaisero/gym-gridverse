import random
import unittest

import gym_gridverse.envs.reset_functions as reset_funcs
from gym_gridverse.geometry import Position, Shape
from gym_gridverse.grid_object import Door, Goal, Key, MovingObstacle, Wall


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


class TestDynamicObstacles(unittest.TestCase):
    def setUp(self):
        self.n = random.randint(1, 3)
        self.w = random.randint(4, 10)
        self.h = random.randint(4, 10)
        self.s = reset_funcs.reset_minigrid_dynamic_obstacles(
            self.h, self.w, self.n
        )

    def test_grid_size(self):
        self.assertEqual(self.s.grid.shape, Shape(self.h, self.w))

    def test_number_obstacles(self):
        count = len(
            list(
                filter(
                    lambda pos: isinstance(self.s.grid[pos], MovingObstacle),
                    self.s.grid.positions(),
                )
            )
        )

        self.assertEqual(count, self.n)


class TestFactory(unittest.TestCase):
    def test_invalid(self):
        self.assertRaises(ValueError, reset_funcs.factory, 'invalid')

    def test_minigrid_empty(self):
        reset_funcs.factory(
            'minigrid_empty', height=10, width=10, random_agent_pos=True
        )
        self.assertRaises(ValueError, reset_funcs.factory, 'minigrid_empty')

    def test_minigrid_four_rooms(self):
        reset_funcs.factory('minigrid_four_rooms', height=10, width=10)
        self.assertRaises(
            ValueError, reset_funcs.factory, 'minigrid_four_rooms'
        )

    def test_minigrid_dynamic_obstacles(self):
        reset_funcs.factory(
            'minigrid_dynamic_obstacles',
            height=10,
            width=10,
            num_obstacles=10,
            random_agent_pos=True,
        )
        self.assertRaises(
            ValueError, reset_funcs.factory, 'minigrid_dynamic_obstacles'
        )

    def test_minigrid_door_key(self):
        reset_funcs.factory(
            'minigrid_door_key', size=10,
        )
        self.assertRaises(ValueError, reset_funcs.factory, 'minigrid_door_key')
