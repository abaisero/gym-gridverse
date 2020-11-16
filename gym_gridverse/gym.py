from functools import partial
from typing import Callable, Dict

import numpy as np

import gym
from gym_gridverse.envs import Environment, factory
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    create_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    create_state_representation,
)


def outer_space_to_gym_space(space: Dict[str, np.ndarray]) -> gym.spaces.Space:
    return gym.spaces.Dict(
        {
            k: gym.spaces.Box(low=np.zeros_like(v), high=v, dtype=np.int)
            for k, v in space.items()
        }
    )


class GymEnvironment(gym.Env):  # pylint: disable=abstract-method
    # NOTE accepting an environment instance as input is a bad idea because it
    # would need to be instantiated during gym registration
    def __init__(self, constructor: Callable[[], OuterEnv]):
        super().__init__()
        self.outer_env = constructor()

        self.state_space = (
            outer_space_to_gym_space(self.outer_env.state_rep.space)
            if self.outer_env.state_rep is not None
            else None
        )
        self.action_space = gym.spaces.Discrete(
            self.outer_env.action_space.num_actions
        )
        self.observation_space = (
            outer_space_to_gym_space(self.outer_env.observation_rep.space)
            if self.outer_env.observation_rep is not None
            else None
        )

    def set_state_representation(self, name: str):
        """Change underlying state representation"""
        self.outer_env.state_rep = create_state_representation(
            name, self.outer_env.inner_env.state_space
        )
        self.state_space = outer_space_to_gym_space(
            self.outer_env.state_rep.space
        )

    def set_observation_representation(self, name: str):
        """Change underlying observation representation"""
        self.outer_env.observation_rep = create_observation_representation(
            name, self.outer_env.inner_env.observation_space
        )
        self.observation_space = outer_space_to_gym_space(
            self.outer_env.observation_rep.space
        )

    @classmethod
    def from_environment(cls, env: OuterEnv):
        return cls(lambda: env)

    @property
    def state(self) -> Dict[str, np.array]:
        return self.outer_env.state

    @property
    def observation(self) -> Dict[str, np.array]:
        return self.outer_env.observation

    def reset(self) -> Dict[str, np.array]:
        self.outer_env.reset()
        return self.observation

    def step(self, action: int):
        action_ = self.outer_env.action_space.int_to_action(action)
        reward, done = self.outer_env.step(action_)
        return self.observation, reward, done, {}

    # TODO implement render method


env_ids = []

for key, constructor_ in factory.STRING_TO_GYM_CONSTRUCTOR.items():

    def outer_env_constructor(
        constructor: Callable[[], Environment]
    ) -> OuterEnv:
        env = constructor()
        state_repr = None
        observation_repr = create_observation_representation(
            'default', env.observation_space
        )
        return OuterEnv(
            env, state_rep=state_repr, observation_rep=observation_repr
        )

    env_id = f'GridVerse-{key}'
    gym.register(
        env_id,
        entry_point='gym_gridverse.gym:GymEnvironment',
        kwargs={'constructor': partial(outer_env_constructor, constructor_)},
    )
    env_ids.append(env_id)
