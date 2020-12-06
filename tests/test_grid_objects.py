""" Tests Grid Object behavior and properties """
import unittest
from typing import Type

import pytest

from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import (
    Box,
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
    factory,
)
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State


class DummyNonRegisteredObject(  # pylint: disable=abstract-method
    GridObject, register=False
):
    """ Some dummy grid object that is _not_ registered """


@pytest.mark.parametrize(
    'object_type,expected',
    [
        (DummyNonRegisteredObject, False),
        (NoneGridObject, True),
        (Hidden, True),
        (Floor, True),
        (Wall, True),
        (Goal, True),
        (Door, True),
        (Key, True),
        (MovingObstacle, True),
        (Box, True),
    ],
)
def test_registration(object_type: Type[GridObject], expected: bool):
    assert (object_type in GridObject.object_types) == expected


def test_none_grid_object_registration():
    """ Tests the registration as a Grid Object """
    assert NoneGridObject in GridObject.object_types


def test_hidden_registration():
    """ Tests the registration as a Grid Object """
    assert Hidden in GridObject.object_types


def test_grid_object_registration():
    """ Test registration of type indices """

    # pylint: disable=no-member
    assert len(GridObject.object_types) == 9
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
            Box.type_index,
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
        Box,
    ]:
        assert GridObject.object_types[obj_cls.type_index] is obj_cls


def simple_state_without_object() -> State:
    """ Returns a 2x2 (empty) grid with an agent without an item """
    return State(
        Grid(height=2, width=2),
        Agent(position=(0, 0), orientation=Orientation.N, obj=Floor()),
    )


def test_none_grid_object_properties():
    """ Basic stupid tests for none grid object """

    none = NoneGridObject()

    assert none.color == Colors.NONE
    assert none.state_index == 0

    assert none.can_be_represented_in_state()
    assert none.render_as_char() == ' '
    assert none.num_states() == 1


def test_hidden_properties():
    """ Basic stupid tests for hidden grid object """

    hidden = Hidden()

    assert not hidden.transparent
    assert hidden.color == Colors.NONE
    assert hidden.state_index == 0

    assert not hidden.can_be_represented_in_state()
    assert hidden.render_as_char() == '.'
    assert hidden.num_states() == 1


def test_floor_properties():
    """ Basic stupid tests for floor grid object """

    floor = Floor()

    assert floor.transparent
    assert not floor.blocks
    assert floor.color == Colors.NONE
    assert not floor.can_be_picked_up
    assert floor.state_index == 0

    assert floor.can_be_represented_in_state()
    assert floor.render_as_char() == ' '
    assert floor.num_states() == 1


def test_wall_properties():
    """ Basic property tests """

    wall = Wall()

    assert not wall.transparent
    assert wall.blocks
    assert wall.color == Colors.NONE
    assert not wall.can_be_picked_up
    assert wall.state_index == 0

    assert wall.can_be_represented_in_state()
    assert wall.render_as_char() == '#'
    assert wall.num_states() == 1


def test_goal_properties():
    """ Basic property tests """

    goal = Goal()

    assert goal.transparent
    assert not goal.blocks
    assert goal.color == Colors.NONE
    assert not goal.can_be_picked_up
    assert goal.state_index == 0

    assert goal.can_be_represented_in_state()
    assert goal.render_as_char() == 'G'
    assert goal.num_states() == 1


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

    assert open_door.can_be_represented_in_state()
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

    assert closed_door.can_be_represented_in_state()
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

    assert locked_door.can_be_represented_in_state()
    assert locked_door.render_as_char() == 'D'


def test_key_properties():
    """ Basic property tests """

    color = Colors.YELLOW
    key = Key(color)

    assert key.transparent
    assert not key.blocks
    assert key.color == color
    assert key.can_be_picked_up
    assert key.state_index == 0

    assert key.can_be_represented_in_state()
    assert key.num_states() == 1


def test_moving_obstacle_basic_properties():
    """Tests basic properties of the moving obstacle"""

    obstacle = MovingObstacle()

    assert obstacle.transparent
    assert not obstacle.blocks
    assert obstacle.color == Colors.NONE
    assert not obstacle.can_be_picked_up
    assert obstacle.state_index == 0

    assert obstacle.can_be_represented_in_state()
    assert obstacle.render_as_char() == '*'
    assert obstacle.num_states() == 1


def test_box_basic_properties():
    """Tests basic properties of box """

    box = Box(Floor())

    assert box.transparent
    assert box.blocks
    assert box.color == Colors.NONE
    assert not box.can_be_picked_up
    assert box.state_index == 0

    assert not box.can_be_represented_in_state()
    assert box.render_as_char() == 'b'
    assert box.num_states() == 1


@pytest.mark.parametrize(
    'name,kwargs',
    [
        ('NoneGridObject', {}),
        ('none_grid_object', {}),
        ('Hidden', {}),
        ('hidden', {}),
        ('Floor', {}),
        ('floor', {}),
        ('Wall', {}),
        ('wall', {}),
        ('Goal', {}),
        ('goal', {}),
        ('Door', {'status': 'LOCKED', 'color': 'RED'}),
        ('door', {'status': 'LOCKED', 'color': 'RED'}),
        ('Key', {'color': 'RED'}),
        ('key', {'color': 'RED'}),
        ('MovingObstacle', {}),
        ('moving_obstacle', {}),
        ('Box', {'obj': Floor()}),
        ('box', {'obj': Floor()}),
    ],
)
def test_factory_valid(name: str, kwargs):
    factory(name, **kwargs)


@pytest.mark.parametrize(
    'name,kwargs,exception',
    [
        ('invalid', {}, ValueError),
        ('Door', {}, ValueError),
        ('door', {}, ValueError),
        ('Key', {}, ValueError),
        ('key', {}, ValueError),
        ('Box', {}, ValueError),
        ('box', {}, ValueError),
    ],
)
def test_factory_invalid(name: str, kwargs, exception: Exception):
    with pytest.raises(exception):  # type: ignore
        factory(name, **kwargs)
