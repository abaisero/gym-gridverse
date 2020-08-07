import unittest

from gym_gridverse.actions import Actions
from gym_gridverse.grid_object import Colors, Door, Floor, Goal
from gym_gridverse.spaces import (
    ActionSpace,
    _max_color_index,
    _max_object_status,
    _max_object_type,
)


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
    def test_contains(self):
        action_space = ActionSpace(list(Actions))
        self.assertTrue(action_space.contains(Actions.MOVE_FORWARD))
        self.assertTrue(action_space.contains(Actions.MOVE_BACKWARD))
        self.assertTrue(action_space.contains(Actions.MOVE_LEFT))
        self.assertTrue(action_space.contains(Actions.MOVE_RIGHT))
        self.assertTrue(action_space.contains(Actions.TURN_LEFT))
        self.assertTrue(action_space.contains(Actions.TURN_RIGHT))
        self.assertTrue(action_space.contains(Actions.ACTUATE))
        self.assertTrue(action_space.contains(Actions.PICK_N_DROP))

        action_space = ActionSpace(
            [
                Actions.MOVE_FORWARD,
                Actions.MOVE_BACKWARD,
                Actions.MOVE_LEFT,
                Actions.MOVE_RIGHT,
            ]
        )
        self.assertTrue(action_space.contains(Actions.MOVE_FORWARD))
        self.assertTrue(action_space.contains(Actions.MOVE_BACKWARD))
        self.assertTrue(action_space.contains(Actions.MOVE_LEFT))
        self.assertTrue(action_space.contains(Actions.MOVE_RIGHT))
        self.assertFalse(action_space.contains(Actions.TURN_LEFT))
        self.assertFalse(action_space.contains(Actions.TURN_RIGHT))
        self.assertFalse(action_space.contains(Actions.ACTUATE))
        self.assertFalse(action_space.contains(Actions.PICK_N_DROP))

    def test_num_actions(self):
        action_space = ActionSpace(list(Actions))
        self.assertEqual(action_space.num_actions, len(Actions))

        action_space = ActionSpace(
            [
                Actions.MOVE_FORWARD,
                Actions.MOVE_BACKWARD,
                Actions.MOVE_LEFT,
                Actions.MOVE_RIGHT,
            ]
        )
        self.assertEqual(action_space.num_actions, 4)
