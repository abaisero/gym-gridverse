from gym_gridverse.action import Action
from gym_gridverse.geometry import Orientation, Position

# maps orientation and action to movement orientation
_move_action_to_orientation = {
    Action.MOVE_FORWARD: Orientation.F,
    Action.MOVE_LEFT: Orientation.L,
    Action.MOVE_RIGHT: Orientation.R,
    Action.MOVE_BACKWARD: Orientation.B,
}


def get_next_position(
    position: Position, orientation: Orientation, action: Action
) -> Position:
    """Returns the tentative next position according to `action`

    NOTE: Assumes successful action and free unobstructed movement.

    Args:
        position (`Position`): current agent position
        orientation (`Orientation`): current agent orientation
        action (`Action`): action taken by agent

    Returns:
        Position: tentative next position
    """

    try:
        move_orientation = _move_action_to_orientation[action]
    except KeyError:
        return position

    return position + Position.from_orientation(orientation * move_orientation)
