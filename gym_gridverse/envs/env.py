import abc
from typing import Optional, Tuple

from gym_gridverse.actions import Actions
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import ActionSpace, ObservationSpace, StateSpace
from gym_gridverse.state import State

__all__ = ['Environment']


class Environment(metaclass=abc.ABCMeta):
    def __init__(
        self,
        state_space: StateSpace,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.observation_space = observation_space

        self._state: Optional[State] = None
        self._observation: Optional[Observation] = None

    @abc.abstractmethod
    def set_seed(self, seed: Optional[int] = None):
        assert False, "Must be implemented by derived class"

    @abc.abstractmethod
    def functional_reset(self) -> State:
        assert False, "Must be implemented by derived class"

    @abc.abstractmethod
    def functional_step(
        self, state: State, action: Actions
    ) -> Tuple[State, float, bool]:
        assert False, "Must be implemented by derived class"

    @abc.abstractmethod
    def functional_observation(self, state: State) -> Observation:
        assert False, "Must be implemented by derived class"

    def reset(self):
        self._state = self.functional_reset()
        self._observation = None

    def step(self, action: Actions) -> Tuple[float, bool]:
        self._state, reward, done = self.functional_step(self.state, action)
        self._observation = None
        return reward, done

    @property
    def state(self) -> State:
        if self._state is None:
            raise RuntimeError(
                'The state was not set properly;  was the environment reset?'
            )

        return self._state

    @property
    def observation(self) -> Observation:
        # memoizing observation because observation function can be stochastic
        if self._observation is None:
            self._observation = self.functional_observation(self.state)

        return self._observation
