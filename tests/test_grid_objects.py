""" Tests Grid Object behavior and properties """
import unittest

import numpy as np
from gym_gridverse.geometry import Position
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Floor,
    Goal,
    GridObject,
    Hidden,
    Key,
    MovingObstacle,
    NoneGridObject,
    Wall,
)
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State


class TestGridObject(unittest.TestCase):
    """ Tests registering Grid Objects """

    def test_no_registration(self):
        """ Grid Objects can be _not_ registered """

        class DummyObject(  # pylint: disable=abstract-method
            GridObject, register=False
        ):
            """ Some dummy grid objet that is _not_ registered """

        self.assertNotIn(DummyObject, GridObject.object_types)

    def test_registration(self):
        """ Test registration of type indices """

        # pylint: disable=no-member
        self.assertEqual(len(GridObject.object_types), 8)
        self.assertCountEqual(
            [
                NoneGridObject.type_index,
                Hidden.type_index,
                Floor.type_index,
                Wall.type_index,
                Goal.type_index,
                Door.type_index,
                Key.type_index,
                MovingObstacle.type_index,
            ],
            range(len(GridObject.object_types)),
        )

        for obj_cls in [
            NoneGridObject,
            Hidden,
            Floor,
            Wall,
            Goal,
            Door,
            Key,
            MovingObstacle,
        ]:
            self.assertIs(GridObject.object_types[obj_cls.type_index], obj_cls)


def simple_state_without_object() -> State:
    """ Returns a 2x2 (empty) grid with an agent without an item

    TODO: Orientation?
    """
    return State(
        Grid(height=2, width=2),
        Agent(position=(0, 0), orientation=None, obj=Floor()),
    )


class TestNoneGridObject(unittest.TestCase):
    """ Basic stupid tests for none grid object """

    def test_registration(self):
        """ Tests the registration as a Grid Object """
        self.assertIn(NoneGridObject, GridObject.object_types)

    def test_properties(self):
        """ Basic stupid tests for none grid object """

        none = NoneGridObject()

        self.assertEqual(none.color, Colors.NONE)
        self.assertEqual(none.state_index, 0)

        expected_arr_represtation = np.array(
            [NoneGridObject.type_index, 0, 0]  # pylint: disable=no-member
        )
        np.testing.assert_array_equal(
            none.as_array(), expected_arr_represtation
        )

        self.assertEqual(none.render_as_char(), ' ')
        self.assertEqual(none.num_states(), 0)


class TestHidden(unittest.TestCase):
    """ Basic stupid tests for floor grid object """

    def test_registration(self):
        """ Tests the registration as a Grid Object """
        self.assertIn(Hidden, GridObject.object_types)

    def test_properties(self):
        """ Basic stupid tests for hidden grid object """

        hidden = Hidden()

        self.assertFalse(hidden.transparent)
        self.assertEqual(hidden.color, Colors.NONE)
        self.assertEqual(hidden.state_index, 0)

        expected_arr_represtation = np.array(
            [Hidden.type_index, 0, 0]  # pylint: disable=no-member
        )
        np.testing.assert_array_equal(
            hidden.as_array(), expected_arr_represtation
        )

        self.assertEqual(hidden.render_as_char(), 'H')

        self.assertEqual(hidden.num_states(), 0)

class TestFloor(unittest.TestCase):
    """ Basic stupid tests for floor grid object """

    def test_registration(self):
        """ Tests the registration as a Grid Object """
        self.assertIn(Floor, GridObject.object_types)

    def test_properties(self):
        """ Basic stupid tests for floor grid object """

        floor = Floor()

        self.assertTrue(floor.transparent)
        self.assertFalse(floor.blocks)
        self.assertEqual(floor.color, Colors.NONE)
        self.assertFalse(floor.can_be_picked_up)
        self.assertEqual(floor.state_index, 0)

        expected_arr_represtation = np.array(
            [Floor.type_index, 0, 0]  # pylint: disable=no-member
        )
        np.testing.assert_array_equal(
            floor.as_array(), expected_arr_represtation
        )

        self.assertEqual(floor.render_as_char(), ' ')
        self.assertEqual(floor.num_states(), 0)


class TestWall(unittest.TestCase):
    """ Basic stupid tests for wall grid object """

    def test_registration(self):
        """ Tests the registration as a Grid Object """
        self.assertIn(Wall, GridObject.object_types)

    def test_properties(self):
        """ Basic property tests """

        wall = Wall()

        self.assertFalse(wall.transparent)
        self.assertTrue(wall.blocks)
        self.assertEqual(wall.color, Colors.NONE)
        self.assertFalse(wall.can_be_picked_up)
        self.assertEqual(wall.state_index, 0)

        expected_arr_represtation = np.array(
            [Wall.type_index, 0, 0]  # pylint: disable=no-member
        )
        np.testing.assert_array_equal(
            wall.as_array(), expected_arr_represtation
        )

        self.assertEqual(wall.render_as_char(), 'W')
        self.assertEqual(wall.num_states(), 0)



class TestGoal(unittest.TestCase):
    """ Basic stupid tests for goal grid object """

    def test_registration(self):
        """ Tests the registration as a Grid Object """
        self.assertIn(Goal, GridObject.object_types)

    def test_properties(self):
        """ Basic property tests """

        goal = Goal()

        self.assertTrue(goal.transparent)
        self.assertFalse(goal.blocks)
        self.assertEqual(goal.color, Colors.NONE)
        self.assertFalse(goal.can_be_picked_up)
        self.assertEqual(goal.state_index, 0)

        expected_arr_represtation = np.array(
            [Goal.type_index, 0, 0]  # pylint: disable=no-member
        )
        np.testing.assert_array_equal(
            goal.as_array(), expected_arr_represtation
        )

        self.assertEqual(goal.render_as_char(), 'G')
        self.assertEqual(goal.num_states(), 0)


class TestDoor(unittest.TestCase):
    """ Tests for the door grid object """

    def test_registration(self):
        """ Tests the registration as a Grid Object """
        self.assertIn(Door, GridObject.object_types)

    def test_open_door_properties(self):
        """ Basic property tests """

        color = Colors.GREEN

        open_door = Door(Door.Status.OPEN, color)

        self.assertTrue(open_door.transparent)
        self.assertEqual(open_door.color, color)
        self.assertFalse(open_door.can_be_picked_up)
        self.assertEqual(open_door.state_index, Door.Status.OPEN.value)
        self.assertTrue(open_door.is_open)
        self.assertFalse(open_door.locked)
        self.assertFalse(open_door.blocks)

        expected_arr_represtation = np.array(
            [Door.type_index, 0, color.value]  # pylint: disable=no-member
        )
        np.testing.assert_array_equal(
            open_door.as_array(), expected_arr_represtation
        )

        self.assertEqual(open_door.render_as_char(), '_')
        self.assertEqual(open_door.num_states(), 3)

    def test_closed_door_properties(self):
        """ Basic property tests """

        color = Colors.NONE

        closed_door = Door(Door.Status.CLOSED, color)

        self.assertFalse(closed_door.transparent)
        self.assertEqual(closed_door.color, color)
        self.assertFalse(closed_door.can_be_picked_up)
        self.assertEqual(closed_door.state_index, Door.Status.CLOSED.value)
        self.assertFalse(closed_door.is_open)
        self.assertFalse(closed_door.locked)
        self.assertTrue(closed_door.blocks)

        expected_arr_represtation = np.array(
            [Door.type_index, 1, color.value]  # pylint: disable=no-member
        )
        np.testing.assert_array_equal(
            closed_door.as_array(), expected_arr_represtation
        )

        self.assertEqual(closed_door.render_as_char(), 'd')

    def test_locked_door_properties(self):
        """ Basic property tests """

        color = Colors.NONE

        locked_door = Door(Door.Status.LOCKED, color)

        self.assertFalse(locked_door.transparent)
        self.assertEqual(locked_door.color, color)
        self.assertFalse(locked_door.can_be_picked_up)
        self.assertEqual(locked_door.state_index, Door.Status.LOCKED.value)
        self.assertFalse(locked_door.is_open)
        self.assertTrue(locked_door.locked)
        self.assertTrue(locked_door.blocks)

        expected_arr_represtation = np.array(
            [Door.type_index, 2, color.value]  # pylint: disable=no-member
        )
        np.testing.assert_array_equal(
            locked_door.as_array(), expected_arr_represtation
        )

        self.assertEqual(locked_door.render_as_char(), 'D')

    def test_opening_door(self):
        """ Testing the simple FNS of the door """

        color = Colors.GREEN
        s = None

        door = Door(Door.Status.CLOSED, color)
        self.assertFalse(door.is_open)

        door.actuate(s)
        self.assertTrue(door.is_open)

        door.actuate(s)
        self.assertTrue(door.is_open)

    def test_opening_door_with_key(self):
        """ Testing the simple FNS of the door """

        color = Colors.BLUE

        # Agent holding wrong key
        s = simple_state_without_object()
        s.agent.obj = Key(Colors.YELLOW)

        door = Door(Door.Status.LOCKED, color)
        self.assertFalse(door.is_open)

        door.actuate(s)
        self.assertFalse(door.is_open)

        s.agent.obj = Key(color)
        self.assertFalse(door.is_open)

        door.actuate(s)
        self.assertTrue(door.is_open)

        door.actuate(s)
        self.assertTrue(door.is_open)


class TestKey(unittest.TestCase):
    """ Basic stupid tests for Key grid object """

    def test_registration(self):
        """ Tests the registration as a Grid Object """
        self.assertIn(Key, GridObject.object_types)

    def test_properties(self):
        """ Basic property tests """

        color = Colors.YELLOW
        key = Key(color)

        self.assertTrue(key.transparent)
        self.assertFalse(key.blocks)
        self.assertEqual(key.color, color)
        self.assertTrue(key.can_be_picked_up)
        self.assertEqual(key.state_index, 0)

        expected_arr_represtation = np.array(
            [Key.type_index, 0, color.value]  # pylint: disable=no-member
        )
        np.testing.assert_array_equal(key.as_array(), expected_arr_represtation)
        self.assertEqual(key.num_states(), 0)


class TestMovingObstacles(unittest.TestCase):
    """Tests the moving obstacles"""

    def test_registration(self):
        """ Tests the registration as a Grid Object """
        self.assertIn(MovingObstacle, GridObject.object_types)

    def test_basic_properties(self):
        """Tests basic properties of the moving obstacle"""

        obstacle = MovingObstacle()

        self.assertTrue(obstacle.transparent)
        self.assertFalse(obstacle.blocks)
        self.assertEqual(obstacle.color, Colors.NONE)
        self.assertFalse(obstacle.can_be_picked_up)
        self.assertEqual(obstacle.state_index, 0)

        expected_arr_represtation = np.array(
            [MovingObstacle.type_index, 0, 0]  # pylint: disable=no-member
        )
        np.testing.assert_array_equal(
            obstacle.as_array(), expected_arr_represtation
        )

        self.assertEqual(obstacle.render_as_char(), '*')
        self.assertEqual(obstacle.num_states(), 0)

    def test_obstacle_movement(self):
        """Test the 'step' behavior of the obstacle"""

        obs_1 = MovingObstacle()
        obs_2 = MovingObstacle()

        # allow for just 1 next step
        s = simple_state_without_object()

        s.grid[Position(1, 0)] = obs_1
        s.grid[Position(1, 1)] = obs_2
        obs_1.step(s, action=None)

        self.assertEqual(s.grid.get_position(obs_1), Position(0, 0))

        # two possible moves out of corner
        s = simple_state_without_object()

        s.grid[Position(1, 1)] = obs_1
        obs_1.step(s, action=None)

        self.assertTrue(
            s.grid.get_position(obs_1) == Position(0, 1)
            or s.grid.get_position(obs_1) == Position(1, 0)
        )


if __name__ == '__main__':
    unittest.main()
