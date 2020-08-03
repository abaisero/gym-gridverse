import abc
import enum
from typing import Tuple

from gym_gridverse.envs.observation_functions import minigrid_observation
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import ActionSpace, ObservationSpace, StateSpace
from gym_gridverse.state import State

__all__ = ['Actions', 'Environment']


class Actions(enum.Enum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = enum.auto()
    MOVE_LEFT = enum.auto()
    MOVE_RIGHT = enum.auto()

    TURN_LEFT = enum.auto()
    TURN_RIGHT = enum.auto()

    ACTUATE = enum.auto()
    PICK_N_DROP = enum.auto()


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

        self.observation_function = minigrid_observation
        self.state = self.functional_reset()

    @abc.abstractmethod
    def functional_reset(self) -> State:
        raise NotImplementedError

    @abc.abstractmethod
    def functional_step(
        self, state: State, action: Actions
    ) -> Tuple[State, float, bool]:
        raise NotImplementedError

    def functional_observation(self, state: State) -> Observation:
        return self.observation_function(state)

    def reset(self):
        self.state = self.functional_reset()
        return self.functional_observation(self.state)

    def step(self, action: Actions) -> Tuple[Observation, float, bool]:
        self.state, reward, done = self.functional_step(self.state, action)
        return (
            self.functional_observation(self.state),
            reward,
            done,
        )
