import unittest

from gym_gridverse.actions import Actions
from gym_gridverse.geometry import Area, Position, Shape
from gym_gridverse.grid_object import Colors, Door, Floor, Goal
from gym_gridverse.spaces import (
    ActionSpace,
    ObservationSpace,
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


class TestObservationSpace(unittest.TestCase):
    def test_area(self):
        observation_space = ObservationSpace(Shape(2, 5), [], [])
        self.assertEqual(observation_space.area, Area((-1, 0), (-2, 2)))

        observation_space = ObservationSpace(Shape(3, 5), [], [])
        self.assertEqual(observation_space.area, Area((-2, 0), (-2, 2)))

        observation_space = ObservationSpace(Shape(2, 7), [], [])
        self.assertEqual(observation_space.area, Area((-1, 0), (-3, 3)))

        observation_space = ObservationSpace(Shape(3, 7), [], [])
        self.assertEqual(observation_space.area, Area((-2, 0), (-3, 3)))

    def test_agent_position(self):
        observation_space = ObservationSpace(Shape(2, 5), [], [])
        self.assertEqual(observation_space.agent_position, Position(1, 2))

        observation_space = ObservationSpace(Shape(3, 5), [], [])
        self.assertEqual(observation_space.agent_position, Position(2, 2))

        observation_space = ObservationSpace(Shape(2, 7), [], [])
        self.assertEqual(observation_space.agent_position, Position(1, 3))

        observation_space = ObservationSpace(Shape(3, 7), [], [])
        self.assertEqual(observation_space.agent_position, Position(2, 3))


if __name__ == '__main__':
    unittest.main()
