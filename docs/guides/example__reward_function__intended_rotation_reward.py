from gym_gridverse.actions import Actions
from gym_gridverse.state import State


def intended_rotation_reward(
    state: State,  # pylint: disable=unused-argument
    action: Actions,
    next_state: State,  # pylint: disable=unused-argument
    *,
    reward_clockwise: float,
    reward_counterclockwise: float,
) -> float:
    """determines reward depending on whether agent has turned (counter)clockwise"""

    if action is Actions.TURN_RIGHT:
        return reward_clockwise

    if action is Actions.TURN_LEFT:
        return reward_counterclockwise

    return 0.0
