import copy
from typing import Optional, Tuple

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.debugging import checkraise
from gym_gridverse.envs import InnerEnv
from gym_gridverse.envs.observation_functions import ObservationFunction
from gym_gridverse.envs.reset_functions import ResetFunction
from gym_gridverse.envs.reward_functions import RewardFunction
from gym_gridverse.envs.terminating_functions import TerminatingFunction
from gym_gridverse.envs.transition_functions import TransitionFunction
from gym_gridverse.observation import Observation
from gym_gridverse.rng import make_rng
from gym_gridverse.spaces import DomainSpace
from gym_gridverse.state import State


class GridWorld(InnerEnv):
    def __init__(
        self,
        domain_space: DomainSpace,
        reset_function: ResetFunction,
        step_function: TransitionFunction,
        observation_function: ObservationFunction,
        reward_function: RewardFunction,
        termination_function: TerminatingFunction,
    ):

        # TODO: maybe add a parameter to avoid calls to `contain` everywhere
        # (or maybe a global setting)

        self._functional_reset = reset_function
        self._functional_step = step_function
        self._functional_observation = observation_function
        self.reward_function = reward_function
        self.termination_function = termination_function

        self._rng: Optional[rnd.Generator] = None

        super().__init__(
            domain_space.state_space,
            domain_space.action_space,
            domain_space.observation_space,
        )

    def set_seed(self, seed: Optional[int] = None):
        self._rng = make_rng(seed)

    def functional_reset(self) -> State:
        state = self._functional_reset(rng=self._rng)
        checkraise(
            lambda: self.state_space.contains(state),
            ValueError,
            'state does not satisfy state-space',
        )

        return state

    def functional_step(
        self, state: State, action: Action
    ) -> Tuple[State, float, bool]:

        checkraise(
            lambda: self.state_space.contains(state),
            ValueError,
            'state does not satisfy state-space',
        )
        checkraise(
            lambda: self.action_space.contains(action),
            ValueError,
            'action {} does not satisfy action-space',
            action,
        )

        next_state = copy.deepcopy(state)
        self._functional_step(next_state, action, rng=self._rng)

        checkraise(
            lambda: self.state_space.contains(next_state),
            ValueError,
            'next_state does not satisfy state-space',
        )

        reward = self.reward_function(state, action, next_state)
        terminal = self.termination_function(state, action, next_state)

        return (next_state, reward, terminal)

    def functional_observation(self, state: State) -> Observation:
        observation = self._functional_observation(state, rng=self._rng)
        checkraise(
            lambda: self.observation_space.contains(observation),
            ValueError,
            'observation does not satisfy observation-space',
        )

        return observation
