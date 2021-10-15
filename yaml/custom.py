from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.envs.transition_functions import update_agent
from gym_gridverse.state import State


#  custom transition function
def multi_update_agent(
    state: State,
    action: Action,
    *,
    n: int,
    rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
):
    for _ in range(n):
        update_agent(state, action)


#  custom reward function
def checkerboard(
    state: State,  # pylint: disable=unused-argument
    action: Action,  # pylint: disable=unused-argument
    next_state: State,
    *,
    reward_even: float,
    reward_odd: float,
):
    y, x = next_state.agent.position.astuple()
    return reward_even if (y + x) % 2 == 0 else reward_odd
