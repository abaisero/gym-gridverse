from gym_gridverse.action import Action
from gym_gridverse.envs.reward_functions import reward_function_registry
from gym_gridverse.state import State


@reward_function_registry.register
def intended_rotation(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_clockwise: float,
    reward_counterclockwise: float,
) -> float:
    """determines reward depending on whether agent has turned (counter)clockwise"""

    return (
        reward_clockwise
        if action is Action.TURN_RIGHT
        else reward_counterclockwise
        if action is Action.TURN_LEFT
        else 0.0
    )
