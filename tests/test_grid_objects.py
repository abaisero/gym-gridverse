""" Tests Grid Object behavior and properties """
import unittest
from typing import Type

import pytest

from gym_gridverse.grid_object import (
    Beacon,
    Box,
    Color,
    Door,
    Exit,
    Floor,
    GridObject,
    Hidden,
    Key,
    MovingObstacle,
    NoneGridObject,
    Telepod,
    Wall,
    grid_object_registry,
)


class DummyNonRegisteredObject(GridObject, register=False):
    """Some dummy grid object that is _not_ registered"""


@pytest.mark.parametrize(
    'object_type,expected',
    [
        (DummyNonRegisteredObject, False),
        (NoneGridObject, True),
        (Hidden, True),
        (Floor, True),
        (Wall, True),
        (Exit, True),
        (Door, True),
        (Key, True),
        (MovingObstacle, True),
        (Box, True),
        (Telepod, True),
        (Beacon, True),
    ],
)
def test_registration(object_type: Type[GridObject], expected: bool):
    assert (object_type in grid_object_registry) == expected


def test_none_grid_object_registration():
    """Tests the registration as a Grid Object"""
    assert NoneGridObject in grid_object_registry


def test_hidden_registration():
    """Tests the registration as a Grid Object"""
    assert Hidden in grid_object_registry


def test_grid_object_registration():
    """Test registration of type indices"""

    assert len(grid_object_registry) == 11
    unittest.TestCase().assertCountEqual(
        [
            NoneGridObject.type_index(),
            Hidden.type_index(),
            Floor.type_index(),
            Wall.type_index(),
            Exit.type_index(),
            Door.type_index(),
            Key.type_index(),
            MovingObstacle.type_index(),
            Box.type_index(),
            Telepod.type_index(),
            Beacon.type_index(),
        ],
        range(len(grid_object_registry)),
    )

    for obj_cls in [
        NoneGridObject,
        Hidden,
        Floor,
        Wall,
        Exit,
        Door,
        Key,
        MovingObstacle,
        Box,
        Telepod,
        Beacon,
    ]:
        assert grid_object_registry[obj_cls.type_index()] is obj_cls


def test_none_grid_object_properties():
    """Basic stupid tests for none grid object"""

    none = NoneGridObject()

    assert none.color == Color.NONE
    assert none.state_index == 0

    assert none.can_be_represented_in_state()
    assert none.num_states() == 1


def test_hidden_properties():
    """Basic stupid tests for hidden grid object"""

    hidden = Hidden()

    assert hidden.blocks_vision
    assert hidden.color == Color.NONE
    assert hidden.state_index == 0

    assert not hidden.can_be_represented_in_state()
    assert hidden.num_states() == 1


def test_floor_properties():
    """Basic stupid tests for floor grid object"""

    floor = Floor()

    assert not floor.blocks_vision
    assert not floor.blocks_movement
    assert floor.color == Color.NONE
    assert not floor.holdable
    assert floor.state_index == 0

    assert floor.can_be_represented_in_state()
    assert floor.num_states() == 1


def test_wall_properties():
    """Basic property tests"""

    wall = Wall()

    assert wall.blocks_vision
    assert wall.blocks_movement
    assert wall.color == Color.NONE
    assert not wall.holdable
    assert wall.state_index == 0

    assert wall.can_be_represented_in_state()
    assert wall.num_states() == 1


def test_exit_properties():
    """Basic property tests"""

    exit_ = Exit()

    assert not exit_.blocks_vision
    assert not exit_.blocks_movement
    assert exit_.color == Color.NONE
    assert not exit_.holdable
    assert exit_.state_index == 0

    assert exit_.can_be_represented_in_state()
    assert exit_.num_states() == 1


def test_door_open_door_properties():
    """Basic property tests"""

    color = Color.GREEN
    open_door = Door(Door.Status.OPEN, color)

    assert not open_door.blocks_vision
    assert open_door.color == color
    assert not open_door.holdable
    assert open_door.state_index == Door.Status.OPEN.value
    assert open_door.is_open
    assert not open_door.is_locked
    assert not open_door.blocks_movement

    assert open_door.can_be_represented_in_state()
    assert open_door.num_states() == 3


def test_door_closed_door_properties():
    """Basic property tests"""

    color = Color.NONE
    closed_door = Door(Door.Status.CLOSED, color)

    assert closed_door.blocks_vision
    assert closed_door.color == color
    assert not closed_door.holdable
    assert closed_door.state_index == Door.Status.CLOSED.value
    assert not closed_door.is_open
    assert not closed_door.is_locked
    assert closed_door.blocks_movement

    assert closed_door.can_be_represented_in_state()


def test_door_locked_door_properties():
    """Basic property tests"""

    color = Color.NONE
    locked_door = Door(Door.Status.LOCKED, color)

    assert locked_door.blocks_vision
    assert locked_door.color == color
    assert not locked_door.holdable
    assert locked_door.state_index == Door.Status.LOCKED.value
    assert not locked_door.is_open
    assert locked_door.is_locked
    assert locked_door.blocks_movement

    assert locked_door.can_be_represented_in_state()


def test_key_properties():
    """Basic property tests"""

    color = Color.YELLOW
    key = Key(color)

    assert not key.blocks_vision
    assert not key.blocks_movement
    assert key.color == color
    assert key.holdable
    assert key.state_index == 0

    assert key.can_be_represented_in_state()
    assert key.num_states() == 1


def test_moving_obstacle_basic_properties():
    """Tests basic properties of the moving obstacle"""

    obstacle = MovingObstacle()

    assert not obstacle.blocks_vision
    assert not obstacle.blocks_movement
    assert obstacle.color == Color.NONE
    assert not obstacle.holdable
    assert obstacle.state_index == 0

    assert obstacle.can_be_represented_in_state()
    assert obstacle.num_states() == 1


def test_box_basic_properties():
    """Tests basic properties of box"""

    box = Box(Floor())

    assert not box.blocks_vision
    assert box.blocks_movement
    assert box.color == Color.NONE
    assert not box.holdable
    assert box.state_index == 0

    assert not box.can_be_represented_in_state()
    assert box.num_states() == 1


def test_telepod_properties():
    """Basic property tests of telepod"""

    color = Color.YELLOW
    telepod = Telepod(color)

    assert not telepod.blocks_vision
    assert not telepod.blocks_movement
    assert telepod.color == color
    assert not telepod.holdable
    assert telepod.state_index == 0

    assert telepod.can_be_represented_in_state()
    assert telepod.num_states() == 1


def test_beacon_properties():
    """Basic property tests of beacon"""

    color = Color.YELLOW
    beacon = Beacon(color)

    assert not beacon.blocks_vision
    assert not beacon.blocks_movement
    assert beacon.color == color
    assert not beacon.holdable
    assert beacon.state_index == 0

    assert beacon.can_be_represented_in_state()
    assert beacon.num_states() == 1


def test_custom_object():
    """Basic property tests of (newly defined) custom objects"""

    class ColoredFloor(GridObject):
        """Most basic _colored_ object in the grid, represents empty cell"""

        state_index = 0
        color = Color.NONE
        blocks_vision = False
        blocks_movement = False
        holdable = False

        def __init__(self, color: Color = Color.NONE):
            self.color = color

        @classmethod
        def can_be_represented_in_state(cls) -> bool:
            return True

        @classmethod
        def num_states(cls) -> int:
            return 1

        def __repr__(self):
            return f'{self.__class__.__name__}({self.color})'

    colored_floor = ColoredFloor(Color.YELLOW)

    assert not colored_floor.blocks_vision
    assert not colored_floor.blocks_movement
    assert colored_floor.color == Color.YELLOW
    assert not colored_floor.holdable
    assert colored_floor.state_index == 0

    assert colored_floor.can_be_represented_in_state()
    assert colored_floor.num_states() == 1

    assert colored_floor.type_index() == len(grid_object_registry) - 1
    assert ColoredFloor.type_index() == len(grid_object_registry) - 1
    assert type(colored_floor) in grid_object_registry
