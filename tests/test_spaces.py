from typing import Sequence, Type

import pytest

from gym_gridverse.actions import Actions
from gym_gridverse.geometry import Area, PositionOrTuple, Shape
from gym_gridverse.grid_object import Colors, Door, Floor, Goal, GridObject
from gym_gridverse.spaces import (
    ActionSpace,
    ObservationSpace,
    _max_color_index,
    _max_object_status,
    _max_object_type,
)


def test_max_color_index():
    colors = [Colors.RED, Colors.BLUE]
    assert _max_color_index(colors) == Colors.BLUE.value


@pytest.mark.parametrize(
    'object_types,expected',
    [([Floor, Goal], 0), ([Door, Goal], len(Door.Status))],
)
def test_max_object_status(
    object_types: Sequence[Type[GridObject]], expected: int
):
    assert _max_object_status(object_types) == expected


@pytest.mark.parametrize(
    'object_types,expected',
    [([Floor, Goal], Goal.type_index), ([Door, Goal], Door.type_index)],
)
def test_max_object_type(
    object_types: Sequence[Type[GridObject]], expected: int
):
    assert _max_object_type(object_types) == expected


@pytest.mark.parametrize(
    'action_space,expected_contains,expected_not_contains',
    [
        (
            ActionSpace(list(Actions)),
            [
                Actions.MOVE_FORWARD,
                Actions.MOVE_BACKWARD,
                Actions.MOVE_LEFT,
                Actions.MOVE_RIGHT,
                Actions.TURN_LEFT,
                Actions.TURN_RIGHT,
                Actions.ACTUATE,
                Actions.PICK_N_DROP,
            ],
            [],
        ),
        (
            ActionSpace(
                [
                    Actions.MOVE_FORWARD,
                    Actions.MOVE_BACKWARD,
                    Actions.MOVE_LEFT,
                    Actions.MOVE_RIGHT,
                ]
            ),
            [
                Actions.MOVE_FORWARD,
                Actions.MOVE_BACKWARD,
                Actions.MOVE_LEFT,
                Actions.MOVE_RIGHT,
            ],
            [
                Actions.TURN_LEFT,
                Actions.TURN_RIGHT,
                Actions.ACTUATE,
                Actions.PICK_N_DROP,
            ],
        ),
    ],
)
def test_action_space_contains(
    action_space: ActionSpace,
    expected_contains: Sequence[Actions],
    expected_not_contains: Sequence[Actions],
):
    for action in expected_contains:
        assert action_space.contains(action)

    for action in expected_not_contains:
        assert not action_space.contains(action)


def test_action_space_num_actions():
    action_space = ActionSpace(list(Actions))
    assert action_space.num_actions == len(Actions)

    action_space = ActionSpace(
        [
            Actions.MOVE_FORWARD,
            Actions.MOVE_BACKWARD,
            Actions.MOVE_LEFT,
            Actions.MOVE_RIGHT,
        ]
    )
    assert action_space.num_actions == 4


@pytest.mark.parametrize(
    'shape,expected',
    [
        (Shape(2, 5), Area((-1, 0), (-2, 2))),
        (Shape(3, 5), Area((-2, 0), (-2, 2))),
        (Shape(2, 7), Area((-1, 0), (-3, 3))),
        (Shape(3, 7), Area((-2, 0), (-3, 3))),
    ],
)
def test_observation_space_area(shape: Shape, expected: Area):
    observation_space = ObservationSpace(shape, [], [])
    assert observation_space.area == expected


@pytest.mark.parametrize(
    'shape,expected',
    [
        (Shape(2, 5), (1, 2)),
        (Shape(3, 5), (2, 2)),
        (Shape(2, 7), (1, 3)),
        (Shape(3, 7), (2, 3)),
    ],
)
def test_observation_space_agent_position(
    shape: Shape, expected: PositionOrTuple
):
    observation_space = ObservationSpace(shape, [], [])
    assert observation_space.agent_position == expected
