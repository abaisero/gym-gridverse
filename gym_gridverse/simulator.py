"""Provides a simulator interface

A simulator is a stateless environment that takes in state-action pairs and
returns the resulting transition

"""

from typing import Optional, Tuple

from gym_gridverse.envs.env import Actions, InnerEnv
from gym_gridverse.observation import Observation
from gym_gridverse.representations.representation import (
    ObservationRepresentation,
    StateRepresentation,
)
from gym_gridverse.state import State


class Simulator:
    """allows for sampling and storing transitions

    A typical usage of this thing is as follows:

    .. code-block:: python

        s, a = ... # assume access to some state and action
        s',o,r,t = simulator.sim(s, a)

        # store data
        data.append(
            simulator.state_rep.convert(s),
            a,
            simulator.state_rep(s'),
            simulator.observation_rep.convert(o),
            r,
            t
        )

        # ... do something with data
    """

    def __init__(
        self,
        env: InnerEnv,
        state_rep: Optional[StateRepresentation] = None,
        observation_rep: Optional[ObservationRepresentation] = None,
    ):
        """creates a simulator from environment and stores state & obs representation

        The representation arguments are optional because they are not
        necessary. However, it is assumed that this class will be used to
        somehow generate transitions, which require the ability to represent
        the transitions, hence this class provides easy access to them
        """
        self.env = env
        self.state_rep = state_rep
        self.observation_rep = observation_rep

    @property
    def action_space(self):
        return self.env.action_space

    def sim(
        self, state: State, action: Actions
    ) -> Tuple[State, Observation, float, bool]:
        """simulates a transition in the environment

        In order to 'use' the states and observation one can use
        `self.state_rep` and `self.observation_rep` to convert them into numpy data
        types

        """
        next_state, reward, terminal = self.env.functional_step(state, action)
        obs = self.env.functional_observation(next_state)

        return next_state, obs, reward, terminal
