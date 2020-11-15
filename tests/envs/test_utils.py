import pytest

from gym_gridverse.actions import Actions
from gym_gridverse.envs.utils import updated_agent_position_if_unobstructed
from gym_gridverse.geometry import Orientation, PositionOrTuple


@pytest.mark.parametrize(
    'position',
    [
        (-5, -5),
        (-5, 0),
        (-5, 5),
        (0, -5),
        (0, 0),
        (0, 5),
        (5, -5),
        (5, 0),
        (5, 5),
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
    position: PositionOrTuple, orientation: Orientation, action: Actions
):
    """ Any action that does not 'move' should not affect next position"""
    assert (
        updated_agent_position_if_unobstructed(position, orientation, action)
        == position
    )


@pytest.mark.parametrize(
    'position,orientation,action,expected',
    [
        ((3, 6), Orientation.N, Actions.MOVE_FORWARD, (2, 6)),
        ((5, 2), Orientation.S, Actions.MOVE_FORWARD, (6, 2)),
        ((1, 2), Orientation.W, Actions.MOVE_BACKWARD, (1, 3)),
        ((4, 1), Orientation.E, Actions.MOVE_LEFT, (3, 1)),
        # off grid
        ((0, 1), Orientation.S, Actions.MOVE_BACKWARD, (-1, 1)),
        ((4, 0), Orientation.N, Actions.MOVE_LEFT, (4, -1)),
    ],
)
def test_basic_moves(
    position: PositionOrTuple,
    orientation: Orientation,
    action: Actions,
    expected: PositionOrTuple,
):
    assert (
        updated_agent_position_if_unobstructed(position, orientation, action)
        == expected
    )
