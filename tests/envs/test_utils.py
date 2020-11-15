import pytest

from gym_gridverse.actions import Actions
from gym_gridverse.envs.utils import updated_agent_position_if_unobstructed
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
        Actions.TURN_LEFT,
        Actions.TURN_RIGHT,
        Actions.ACTUATE,
        Actions.PICK_N_DROP,
    ],
)
def test_non_movement_actions(
    position: Position, orientation: Orientation, action: Actions
):
    """ Any action that does not 'move' should not affect next position"""
    assert (
        updated_agent_position_if_unobstructed(position, orientation, action)
        == position
    )


@pytest.mark.parametrize(
    'position,orientation,action,expected',
    [
        (Position(3, 6), Orientation.N, Actions.MOVE_FORWARD, Position(2, 6)),
        (Position(5, 2), Orientation.S, Actions.MOVE_FORWARD, Position(6, 2)),
        (Position(1, 2), Orientation.W, Actions.MOVE_BACKWARD, Position(1, 3)),
        (Position(4, 1), Orientation.E, Actions.MOVE_LEFT, Position(3, 1)),
        # off grid
        (Position(0, 1), Orientation.S, Actions.MOVE_BACKWARD, Position(-1, 1)),
        (Position(4, 0), Orientation.N, Actions.MOVE_LEFT, Position(4, -1)),
    ],
)
def test_basic_moves(
    position: Position,
    orientation: Orientation,
    action: Actions,
    expected: Position,
):
    assert (
        updated_agent_position_if_unobstructed(position, orientation, action)
        == expected
    )
