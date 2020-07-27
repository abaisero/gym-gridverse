import abc
import enum
from typing import Dict, Optional, Tuple

from .spaces import (
    ActionSpace,
    Observation,
    ObservationSpace,
    State,
    StateSpace,
)


class Actions(enum.Enum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = enum.auto()
    MOVE_LEFT = enum.auto()
    MOVE_RIGHT = enum.auto()

    TURN_LEFT = enum.auto()
    TURN_RIGHT = enum.auto()

    ACTUATE = enum.auto()

    PICK = enum.auto()
    DROP = enum.auto()


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

        self.state: Optional[State] = None

    @abc.abstractmethod
    def functional_reset(self) -> State:
        raise NotImplementedError

    @abc.abstractmethod
    def functional_step(
        self, state: State, action: int
    ) -> Tuple[Observation, float, bool, Dict]:
        raise NotImplementedError

    def functional_observation(self, state: State) -> Observation:
        raise NotImplementedError

    @property
    def observation(self):
        return self.functional_observation(self.state)

    def reset(self):
        self.state = self.functional_reset()
        return self.observation

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict]:
        self.state, reward, done, info = self.functional_step(
            self.state, action
        )
        return self.observation, reward, done, info


class CompactEnvironment(Environment):
    raise NotImplementedError
