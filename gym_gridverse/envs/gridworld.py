import copy
from typing import Optional, Tuple

import numpy.random as rnd

from gym_gridverse.actions import Actions
from gym_gridverse.envs import Environment
from gym_gridverse.envs.observation_functions import ObservationFunction
from gym_gridverse.envs.reset_functions import ResetFunction
from gym_gridverse.envs.reward_functions import RewardFunction
from gym_gridverse.envs.terminating_functions import TerminatingFunction
from gym_gridverse.envs.transition_functions import TransitionFunction
from gym_gridverse.observation import Observation
from gym_gridverse.rng import make_rng
from gym_gridverse.spaces import DomainSpace
from gym_gridverse.state import State


class GridWorld(Environment):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        domain_space: DomainSpace,
        reset_function: ResetFunction,
        step_function: TransitionFunction,
        observation_function: ObservationFunction,
        reward_function: RewardFunction,
        termination_function: TerminatingFunction,
    ):

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

    def set_observation_function(
        self, observation_function: ObservationFunction
    ):
        self._functional_observation = observation_function
        # in `Environment`: set to make sure next call to observation will
        # actually generate a new one
        self._observation = None

    def functional_reset(self) -> State:
        state = self._functional_reset(rng=self._rng)
        if not self.state_space.contains(state):
            raise ValueError('state does not satisfy state-space')

        return state

    def functional_step(
        self, state: State, action: Actions
    ) -> Tuple[State, float, bool]:

        if not self.state_space.contains(state):
            raise ValueError('state does not satisfy state-space')

        if not self.action_space.contains(action):
            raise ValueError(f'action {action} does not satisfy action-space')

        next_state = copy.deepcopy(state)
        self._functional_step(next_state, action, rng=self._rng)

        if not self.state_space.contains(next_state):
            raise ValueError('next_state does not satisfy state-space')

        reward = self.reward_function(state, action, next_state)
        terminal = self.termination_function(state, action, next_state)

        return (next_state, reward, terminal)

    def functional_observation(self, state: State) -> Observation:
        observation = self._functional_observation(state, rng=self._rng)
        if not self.observation_space.contains(observation):
            raise ValueError('observation does not satisfy observation-space')

        return observation
