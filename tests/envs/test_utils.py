import pytest

from gym_gridverse.action import Action
from gym_gridverse.envs.utils import get_next_position
from gym_gridverse.geometry import Orientation, Position


@pytest.mark.parametrize(
    'position',
    [
        Position(-5, -5),
        Position(-5, 0),
        Position(-5, 5),
        Position(0, -5),
        Position(0, 0),
        Position(0, 5),
        Position(5, -5),
        Position(5, 0),
        Position(5, 5),
    ],
)
@pytest.mark.parametrize('orientation', list(Orientation))
@pytest.mark.parametrize(
    'action',
    [
        Action.TURN_LEFT,
        Action.TURN_RIGHT,
        Action.ACTUATE,
        Action.PICK_N_DROP,
    ],
)
def test_non_movement_actions(
    position: Position, orientation: Orientation, action: Action
):
    """Any action that does not 'move' should not affect next position"""
    assert get_next_position(position, orientation, action) == position


@pytest.mark.parametrize(
    'position,orientation,action,expected',
    [
        (
            Position(3, 6),
            Orientation.F,
            Action.MOVE_FORWARD,
            Position(2, 6),
        ),
        (
            Position(5, 2),
            Orientation.B,
            Action.MOVE_FORWARD,
            Position(6, 2),
        ),
        (
            Position(1, 2),
            Orientation.L,
            Action.MOVE_BACKWARD,
            Position(1, 3),
        ),
        (Position(4, 1), Orientation.R, Action.MOVE_LEFT, Position(3, 1)),
        # off grid
        (
            Position(0, 1),
            Orientation.B,
            Action.MOVE_BACKWARD,
            Position(-1, 1),
        ),
        (
            Position(4, 0),
            Orientation.F,
            Action.MOVE_LEFT,
            Position(4, -1),
        ),
    ],
)
def test_basic_moves(
    position: Position,
    orientation: Orientation,
    action: Action,
    expected: Position,
):
    assert get_next_position(position, orientation, action) == expected
