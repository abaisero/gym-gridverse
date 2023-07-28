from typing import Optional, Tuple

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.debugging import gv_debug
from gym_gridverse.envs import InnerEnv
from gym_gridverse.envs.observation_functions import ObservationFunction
from gym_gridverse.envs.reset_functions import ResetFunction
from gym_gridverse.envs.reward_functions import RewardFunction
from gym_gridverse.envs.terminating_functions import TerminatingFunction
from gym_gridverse.envs.transition_functions import (
    TransitionFunction,
    transition_with_copy,
)
from gym_gridverse.observation import Observation
from gym_gridverse.rng import make_rng
from gym_gridverse.spaces import ActionSpace, ObservationSpace, StateSpace
from gym_gridverse.state import State


class GridWorld(InnerEnv):
    """Implementation of the InnerEnv interface."""

    def __init__(
        self,
        state_space: StateSpace,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        reset_function: ResetFunction,
        transition_function: TransitionFunction,
        observation_function: ObservationFunction,
        reward_function: RewardFunction,
        termination_function: TerminatingFunction,
    ):
        """Initializes a GridWorld from the given components.

        Args:
            state_space (StateSpace):
            action_space (ActionSpace):
            observation_space (ObservationSpace):
            reset_function: (ResetFunction):
            transition_function: (TransitionFunction),:
            observation_function (ObservationFunction):
            reward_function (RewardFunction):
            termination_function (TerminatingFunction):
        """

        # TODO: maybe add a parameter to avoid calls to `contain` everywhere
        # (or maybe a global setting)

        self._reset_function = reset_function
        self._transition_function = transition_function
        self._observation_function = observation_function
        self._reward_function = reward_function
        self._termination_function = termination_function

        self._rng: Optional[rnd.Generator] = None

        super().__init__(state_space, action_space, observation_space)

    def set_seed(self, seed: Optional[int] = None):
        self._rng = make_rng(seed)

    def functional_reset(self) -> State:
        state = self._reset_function(rng=self._rng)
        if gv_debug() and not self.state_space.contains(state):
            raise ValueError('state does not satisfy state_space')

        return state

    def functional_step(
        self, state: State, action: Action
    ) -> Tuple[State, float, bool]:
        if gv_debug() and not self.state_space.contains(state):
            raise ValueError('state does not satisfy state_space')
        if not self.action_space.contains(action):
            raise ValueError('action {action} does not satisfy action-space')

        next_state = transition_with_copy(
            self._transition_function,
            state,
            action,
            rng=self._rng,
        )

        if gv_debug() and not self.state_space.contains(next_state):
            raise ValueError('next_state does not satisfy state_space')

        reward = self._reward_function(state, action, next_state)
        terminal = self._termination_function(state, action, next_state)

        return (next_state, reward, terminal)

    def functional_observation(self, state: State) -> Observation:
        observation = self._observation_function(state, rng=self._rng)
        if gv_debug() and not self.observation_space.contains(observation):
            raise ValueError('observation does not satisfy observation_space')

        return observation
