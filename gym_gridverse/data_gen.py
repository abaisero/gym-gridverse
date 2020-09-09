""" Collection of API functions for generation environment interactions

Of interest of those looking to do model learning, this module provides ways of
generating data to do so

"""

from typing import Callable, Dict, List, NamedTuple

import numpy as np

from gym_gridverse.envs.env import Actions
from gym_gridverse.simulator import Simulator
from gym_gridverse.state import State


class Transition(NamedTuple):
    """s,a -> s,o,r,t transition """

    state: Dict[str, np.ndarray]
    action: Actions
    next_state: Dict[str, np.ndarray]
    obs: Dict[str, np.ndarray]
    reward: float
    terminal: bool


def sample_transitions(
    n: int,
    state_sampler: Callable[[], State],
    action_sampler: Callable[[State], Actions],
    sim: Simulator,
) -> List[Transition]:
    """samples transitions uniformly

    Returns a list of `n` `Transition`, where the state-action pair is sampled
    from `state_sampler` and `action_sampler`, and the resulting transition
    according to `sim`. The states and observation is encoded as numpy arrays
    through the representation stored in `sim`.

    Assumes: the `state_rep` and `obs_rep` in `sim` is set (not None)

    Args:
        n (`int`): number of sampled transitions
        state_sampler (`Callable[[], State]`): input state sample method
        action_sampler (`Callable[[State], Actions]`): input action sample method
        sim (`Simulator`): simulator

    Returns:
        `List[Transition]`: list of `Transition` of size `n`
    """
    # this function assumes those representations are available
    assert sim.state_rep
    assert sim.obs_rep

    data: List[Transition] = []

    for _ in range(n):

        # uniformly sample state-action pair
        s = state_sampler()
        a = action_sampler(s)

        # simulate transition
        next_s, o, r, t = sim.sim(s, a)

        # store (encoded) transition
        data.append(
            Transition(
                sim.state_rep.convert(s),
                a,
                sim.state_rep.convert(next_s),
                sim.obs_rep.convert(o),
                r,
                t,
            )
        )

    return data
