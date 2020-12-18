from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.envs.transition_functions import TransitionFunction
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State


def random_transition(
    state: State,
    action: Action,
    *,
    transition_function: TransitionFunction,
    p_success: float,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """randomly determines whether to perform a transition or not"""

    rng = get_gv_rng_if_none(rng)  # necessary to use rng object!

    # flip coin to see whether to run the transition_function
    success = rng.random() <= p_success
    if success:
        transition_function(state, action, rng=rng)
