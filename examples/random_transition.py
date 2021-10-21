from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.envs.transition_functions import (
    TransitionFunction,
    transition_function_registry,
)
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State


@transition_function_registry.register
def random_transition(
    state: State,
    action: Action,
    *,
    transition_function: TransitionFunction,
    p_success: float,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """randomly determines whether to perform a transition"""

    rng = get_gv_rng_if_none(rng)  # necessary to use rng object!

    # flip coin to run the transition_function
    if rng.random() <= p_success:
        transition_function(state, action, rng=rng)
