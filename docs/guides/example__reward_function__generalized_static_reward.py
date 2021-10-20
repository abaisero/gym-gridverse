import functools

from gym_gridverse.action import Action
from gym_gridverse.state import State


def generalized_static_reward(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_if_equals: float = -1.0,
    reward_if_not_equals: float = 0.0,
) -> float:
    """determines reward depending on whether state is unchanged"""
    return reward_if_equals if state == next_state else reward_if_not_equals


# binding two variants of generalized_static_reward
stronger_static_reward = functools.partial(
    generalized_static_reward, reward_if_equals=-2.0, reward_if_not_equals=1.0
)
weaker_static_reward = functools.partial(
    generalized_static_reward, reward_if_equale=-0.2, reward_if_not_equals=0.1
)
