from gym_gridverse.action import Action
from gym_gridverse.geometry import Orientation, Position

# maps orientation and action to position delta
_delta_dict = {
    **{
        (orientation, Action.MOVE_FORWARD): orientation.as_position()
        for orientation in Orientation
    },
    **{
        (orientation, Action.MOVE_LEFT): orientation.rotate_left().as_position()
        for orientation in Orientation
    },
    **{
        (
            orientation,
            Action.MOVE_RIGHT,
        ): orientation.rotate_right().as_position()
        for orientation in Orientation
    },
    **{
        (
            orientation,
            Action.MOVE_BACKWARD,
        ): orientation.rotate_back().as_position()
        for orientation in Orientation
    },
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

    return position + _delta_dict.get((orientation, action), Position(0, 0))
