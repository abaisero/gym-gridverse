from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.envs.reward_functions import reward_function_registry
from gym_gridverse.envs.transition_functions import (
    transition_function_registry,
    update_agent,
)
from gym_gridverse.state import State


#  custom transition function
@transition_function_registry.register
def multi_update_agent(
    state: State,
    action: Action,
    *,
    n: int,
    rng: Optional[rnd.Generator] = None,
):
    for _ in range(n):
        update_agent(state, action)


#  custom reward function
@reward_function_registry.register
def checkerboard(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_even: float,
    reward_odd: float,
):
    return (
        reward_even
        if (next_state.agent.position.y + next_state.agent.position.x) % 2 == 0
        else reward_odd
    )
