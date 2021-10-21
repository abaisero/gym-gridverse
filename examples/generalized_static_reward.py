import functools

from gym_gridverse.action import Action
from gym_gridverse.envs.reward_functions import reward_function_registry
from gym_gridverse.state import State


@reward_function_registry.register
def generalized_static(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_if_static: float = -1.0,
    reward_if_not_static: float = 0.0,
) -> float:
    """determines reward depending on whether state is unchanged"""
    return reward_if_static if state == next_state else reward_if_not_static


# binding two variants of generalized_static_reward
stronger_static = functools.partial(
    generalized_static,
    reward_if_static=-2.0,
    reward_if_not_static=1.0,
)
weaker_static = functools.partial(
    generalized_static,
    reward_if_static=-0.2,
    reward_if_not_static=0.1,
)
