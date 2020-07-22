import abc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gym
import numpy as np


@dataclass
class RLData:
    """Dataclass to represent states and observations"""

    grid: np.ndarray
    array: np.ndarray


State = RLData
Observation = RLData
Feedback = Tuple[Observation, float, bool, Dict]


class Environment(gym.Env, metaclass=abc.ABCMeta):
    """Environment"""

    def __init__(self):
        super().__init__()
        self.state: Optional[State] = None

    @abc.abstractmethod
    def functional_reset(self) -> State:
        raise NotImplementedError

    @abc.abstractmethod
    def functional_step(self, state: State, action: int) -> Feedback:
        raise NotImplementedError

    def functional_observation(self, state: State) -> Observation:
        observation_grid = ...  # TODO implement slicing and observation-masking
        observation_array = state.array
        return Observation(grid=observation_grid, array=observation_array)

    @property
    def observation(self):
        return self.functional_observation(self.state)

    def reset(self):
        self.state = self.functional_reset()
        return self.observation

    def step(self, action: int) -> Feedback:
        self.state, reward, done, info = self.functional_step(
            self.state, action
        )
        return self.observation, reward, done, info
