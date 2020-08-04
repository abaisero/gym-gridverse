import unittest

from gym_gridverse.envs import Actions
from gym_gridverse.grid_object import Colors, Door, Floor, Goal
from gym_gridverse.spaces import (ActionSpace, _max_color_index,
                                  _max_object_status, _max_object_type)


class TestGetMaxValues(unittest.TestCase):
    def test_max_color_index(self):
        colors = [Colors.RED, Colors.BLUE]
        self.assertEqual(_max_color_index(colors), Colors.BLUE.value)

    def test_max_object_status(self):

        self.assertEqual(_max_object_status([Floor, Goal]), 0)
        self.assertEqual(_max_object_status([Goal, Door]), len(Door.Status))

    def test_max_object_type(self):
        self.assertEqual(_max_object_type([Floor, Goal]), Goal.type_index)
        self.assertEqual(_max_object_type([Door, Goal]), Door.type_index)


class TestActionSpace(unittest.TestCase):
    def test_num_actions(self):
        self.assertEqual(len(Actions), ActionSpace().num_actions)
