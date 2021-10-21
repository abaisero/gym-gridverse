from gym_gridverse.action import Action
from gym_gridverse.envs.reward_functions import reward_function_registry
from gym_gridverse.state import State


@reward_function_registry.register
def static(
    state: State,
    action: Action,
    next_state: State,
) -> float:
    """negative reward if state is unchanged"""
    return -1.0 if state == next_state else 0.0
