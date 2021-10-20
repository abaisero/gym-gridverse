from gym_gridverse.action import Action
from gym_gridverse.state import State


def intended_rotation_reward(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_clockwise: float,
    reward_counterclockwise: float,
) -> float:
    """determines reward depending on whether agent has turned (counter)clockwise"""

    if action is Action.TURN_RIGHT:
        return reward_clockwise

    if action is Action.TURN_LEFT:
        return reward_counterclockwise

    return 0.0
