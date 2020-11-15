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


def test_grid_object_no_registration():
    """ Grid Objects can be _not_ registered """

    class DummyObject(  # pylint: disable=abstract-method
        GridObject, register=False
    ):
        """ Some dummy grid objet that is _not_ registered """

    assert DummyObject not in GridObject.object_types


def test_grid_object_registration():
    """ Test registration of type indices """

    # pylint: disable=no-member
    assert len(GridObject.object_types) == 8
    unittest.TestCase().assertCountEqual(
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
        assert GridObject.object_types[obj_cls.type_index] is obj_cls


def simple_state_without_object() -> State:
    """ Returns a 2x2 (empty) grid with an agent without an item """
    return State(
        Grid(height=2, width=2),
        Agent(position=(0, 0), orientation=None, obj=Floor()),
    )


def test_none_grid_object_registration():
    """ Tests the registration as a Grid Object """
    assert NoneGridObject in GridObject.object_types


def test_none_grid_object_properties():
    """ Basic stupid tests for none grid object """

    none = NoneGridObject()

    assert none.color == Colors.NONE
    assert none.state_index == 0

    expected_arr_represtation = np.array(
        [NoneGridObject.type_index, 0, 0]  # pylint: disable=no-member
    )
    np.testing.assert_array_equal(none.as_array(), expected_arr_represtation)

    assert none.render_as_char() == ' '
    assert none.num_states() == 0


def test_hidden_registration():
    """ Tests the registration as a Grid Object """
    assert Hidden in GridObject.object_types


def test_hidden_properties():
    """ Basic stupid tests for hidden grid object """

    hidden = Hidden()

    assert not hidden.transparent
    assert hidden.color == Colors.NONE
    assert hidden.state_index == 0

    expected_arr_represtation = np.array(
        [Hidden.type_index, 0, 0]  # pylint: disable=no-member
    )
    np.testing.assert_array_equal(hidden.as_array(), expected_arr_represtation)

    assert hidden.render_as_char() == '.'

    assert hidden.num_states() == 0


def test_floor_registration():
    """ Tests the registration as a Grid Object """
    assert Floor in GridObject.object_types


def test_floor_properties():
    """ Basic stupid tests for floor grid object """

    floor = Floor()

    assert floor.transparent
    assert not floor.blocks
    assert floor.color == Colors.NONE
    assert not floor.can_be_picked_up
    assert floor.state_index == 0

    expected_arr_represtation = np.array(
        [Floor.type_index, 0, 0]  # pylint: disable=no-member
    )
    np.testing.assert_array_equal(floor.as_array(), expected_arr_represtation)

    assert floor.render_as_char() == ' '
    assert floor.num_states() == 0


def test_wall_registration():
    """ Tests the registration as a Grid Object """
    assert Wall in GridObject.object_types


def test_wall_properties():
    """ Basic property tests """

    wall = Wall()

    assert not wall.transparent
    assert wall.blocks
    assert wall.color == Colors.NONE
    assert not wall.can_be_picked_up
    assert wall.state_index == 0

    expected_arr_represtation = np.array(
        [Wall.type_index, 0, 0]  # pylint: disable=no-member
    )
    np.testing.assert_array_equal(wall.as_array(), expected_arr_represtation)

    assert wall.render_as_char() == '#'
    assert wall.num_states() == 0


def test_goal_registration():
    """ Tests the registration as a Grid Object """
    assert Goal in GridObject.object_types


def test_goal_properties():
    """ Basic property tests """

    goal = Goal()

    assert goal.transparent
    assert not goal.blocks
    assert goal.color == Colors.NONE
    assert not goal.can_be_picked_up
    assert goal.state_index == 0

    expected_arr_represtation = np.array(
        [Goal.type_index, 0, 0]  # pylint: disable=no-member
    )
    np.testing.assert_array_equal(goal.as_array(), expected_arr_represtation)

    assert goal.render_as_char() == 'G'
    assert goal.num_states() == 0


def test_door_registration():
    """ Tests the registration as a Grid Object """
    assert Door in GridObject.object_types


def test_door_open_door_properties():
    """ Basic property tests """

    color = Colors.GREEN

    open_door = Door(Door.Status.OPEN, color)

    assert open_door.transparent
    assert open_door.color == color
    assert not open_door.can_be_picked_up
    assert open_door.state_index == Door.Status.OPEN.value
    assert open_door.is_open
    assert not open_door.locked
    assert not open_door.blocks

    expected_arr_represtation = np.array(
        [Door.type_index, 0, color.value]  # pylint: disable=no-member
    )
    np.testing.assert_array_equal(
        open_door.as_array(), expected_arr_represtation
    )

    assert open_door.render_as_char() == '_'
    assert open_door.num_states() == 3


def test_door_closed_door_properties():
    """ Basic property tests """

    color = Colors.NONE

    closed_door = Door(Door.Status.CLOSED, color)

    assert not closed_door.transparent
    assert closed_door.color == color
    assert not closed_door.can_be_picked_up
    assert closed_door.state_index == Door.Status.CLOSED.value
    assert not closed_door.is_open
    assert not closed_door.locked
    assert closed_door.blocks

    expected_arr_represtation = np.array(
        [Door.type_index, 1, color.value]  # pylint: disable=no-member
    )
    np.testing.assert_array_equal(
        closed_door.as_array(), expected_arr_represtation
    )

    assert closed_door.render_as_char() == 'd'


def test_door_locked_door_properties():
    """ Basic property tests """

    color = Colors.NONE

    locked_door = Door(Door.Status.LOCKED, color)

    assert not locked_door.transparent
    assert locked_door.color == color
    assert not locked_door.can_be_picked_up
    assert locked_door.state_index == Door.Status.LOCKED.value
    assert not locked_door.is_open
    assert locked_door.locked
    assert locked_door.blocks

    expected_arr_represtation = np.array(
        [Door.type_index, 2, color.value]  # pylint: disable=no-member
    )
    np.testing.assert_array_equal(
        locked_door.as_array(), expected_arr_represtation
    )

    assert locked_door.render_as_char() == 'D'


def test_door_opening_door():
    """ Testing the simple FNS of the door """

    color = Colors.GREEN
    s = None

    door = Door(Door.Status.CLOSED, color)
    assert not door.is_open

    door.actuate(s)
    assert door.is_open

    door.actuate(s)
    assert door.is_open


def test_door_opening_door_with_key():
    """ Testing the simple FNS of the door """

    color = Colors.BLUE

    # Agent holding wrong key
    s = simple_state_without_object()
    s.agent.obj = Key(Colors.YELLOW)

    door = Door(Door.Status.LOCKED, color)
    assert not door.is_open

    door.actuate(s)
    assert not door.is_open

    s.agent.obj = Key(color)
    assert not door.is_open

    door.actuate(s)
    assert door.is_open

    door.actuate(s)
    assert door.is_open


def test_key_registration():
    """ Tests the registration as a Grid Object """
    assert Key in GridObject.object_types


def test_key_properties():
    """ Basic property tests """

    color = Colors.YELLOW
    key = Key(color)

    assert key.transparent
    assert not key.blocks
    assert key.color == color
    assert key.can_be_picked_up
    assert key.state_index == 0

    expected_arr_represtation = np.array(
        [Key.type_index, 0, color.value]  # pylint: disable=no-member
    )
    np.testing.assert_array_equal(key.as_array(), expected_arr_represtation)
    assert key.num_states() == 0


def test_moving_obstacle_registration():
    """ Tests the registration as a Grid Object """
    assert MovingObstacle in GridObject.object_types


def test_moving_obstacle_basic_properties():
    """Tests basic properties of the moving obstacle"""

    obstacle = MovingObstacle()

    assert obstacle.transparent
    assert not obstacle.blocks
    assert obstacle.color == Colors.NONE
    assert not obstacle.can_be_picked_up
    assert obstacle.state_index == 0

    expected_arr_represtation = np.array(
        [MovingObstacle.type_index, 0, 0]  # pylint: disable=no-member
    )
    np.testing.assert_array_equal(
        obstacle.as_array(), expected_arr_represtation
    )

    assert obstacle.render_as_char() == '*'
    assert obstacle.num_states() == 0


def test_moving_obstacle_obstacle_movement():
    """Test the 'step' behavior of the obstacle"""

    obs_1 = MovingObstacle()
    obs_2 = MovingObstacle()

    # allow for just 1 next step
    s = simple_state_without_object()

    s.grid[Position(1, 0)] = obs_1
    s.grid[Position(1, 1)] = obs_2
    obs_1.step(s, action=None)

    assert s.grid.get_position(obs_1) == Position(0, 0)

    # two possible moves out of corner
    s = simple_state_without_object()

    s.grid[Position(1, 1)] = obs_1
    obs_1.step(s, action=None)

    assert (s.grid.get_position(obs_1) == Position(0, 1)) or (
        s.grid.get_position(obs_1) == Position(1, 0)
    )
