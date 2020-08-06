from typing import Callable, Dict

import numpy as np

import gym
from gym_gridverse.envs import Environment, factory


class GymGridVerseEnvironment(gym.Env):  # pylint: disable=abstract-method
    # NOTE accepting an environment instance as input is a bad idea because it
    # would need to be instantiated during gym registration
    def __init__(self, constructor: Callable[[], Environment]):
        super().__init__()
        self.env = constructor()

        self.state_space = gym.spaces.Dict(
            {
                'grid': gym.spaces.Box(self.env.state_space.grid.blarg),
                'agent': gym.spaces.Box(self.env.state_space.agent.blarg),
            }
        )
        self.action_space = gym.spaces.Discrete(
            self.env.action_space.num_actions
        )
        self.observation_space = gym.spaces.Dict(
            {
                'grid': gym.spaces.Box(self.env.observation_space.grid.blarg),
                'agent': gym.spaces.Box(self.env.observation_space.agent.blarg),
            }
        )

    @classmethod
    def from_environment(cls, env: Environment):
        return cls(lambda: env)

    @property
    def state(self) -> Dict[str, np.array]:
        return {
            'grid': self.env.state.grid.as_array(),
            'agent': self.env.state.agent.as_array(),
        }

    @property
    def observation(self) -> Dict[str, np.array]:
        return {
            'grid': self.env.observation.grid.as_array(),
            'agent': self.env.observation.agent.as_array(),
        }

    def reset(self) -> Dict[str, np.array]:
        self.env.reset()
        return self.observation

    def step(self, action: int):
        action_ = self.env.action_space.int_to_action(action)
        reward, done = self.env.step(action_)
        return self.observation, reward, done, {}


for key, constructor_ in factory.STRING_TO_GYM_CONSTRUCTOR.items():
    gym.register(
        f'GridVerse-{key}',
        entry_point='gym_gridverse.gym:GymGridVerseEnvironment',
        kwargs={'constructor': constructor_},
    )
