import abc
from typing import Tuple

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

        self._state: State

    @abc.abstractmethod
    def functional_reset(self) -> State:
        raise NotImplementedError

    @abc.abstractmethod
    def functional_step(
        self, state: State, action: Actions
    ) -> Tuple[State, float, bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def functional_observation(self, state: State) -> Observation:
        raise NotImplementedError

    def reset(self):
        self.state = self.functional_reset()

    def step(self, action: Actions) -> Tuple[float, bool]:
        self.state, reward, done = self.functional_step(self.state, action)
        return reward, done

    @property
    def state(self):
        try:
            return self._state
        except AttributeError:
            raise RuntimeError(
                'The state was not set;  check that the environment was reset.'
            )

    @state.setter
    def state(self, value: State):
        self._state = value

    @property
    def observation(self):
        return self.functional_observation(self.state)
