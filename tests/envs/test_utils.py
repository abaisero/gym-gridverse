import pytest

from gym_gridverse.action import Action
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
        Action.TURN_LEFT,
        Action.TURN_RIGHT,
        Action.ACTUATE,
        Action.PICK_N_DROP,
    ],
)
def test_non_movement_actions(
    position: PositionOrTuple, orientation: Orientation, action: Action
):
    """Any action that does not 'move' should not affect next position"""
    assert (
        updated_agent_position_if_unobstructed(position, orientation, action)
        == position
    )


@pytest.mark.parametrize(
    'position,orientation,action,expected',
    [
        ((3, 6), Orientation.N, Action.MOVE_FORWARD, (2, 6)),
        ((5, 2), Orientation.S, Action.MOVE_FORWARD, (6, 2)),
        ((1, 2), Orientation.W, Action.MOVE_BACKWARD, (1, 3)),
        ((4, 1), Orientation.E, Action.MOVE_LEFT, (3, 1)),
        # off grid
        ((0, 1), Orientation.S, Action.MOVE_BACKWARD, (-1, 1)),
        ((4, 0), Orientation.N, Action.MOVE_LEFT, (4, -1)),
    ],
)
def test_basic_moves(
    position: PositionOrTuple,
    orientation: Orientation,
    action: Action,
    expected: PositionOrTuple,
):
    assert (
        updated_agent_position_if_unobstructed(position, orientation, action)
        == expected
    )
