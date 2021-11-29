from __future__ import annotations

import time
from functools import partial
from typing import Callable, Dict, List, Optional

import gym
import numpy as np
from gym.utils import seeding

from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    create_observation_representation,
)
from gym_gridverse.representations.spaces import Space, SpaceType
from gym_gridverse.representations.state_representations import (
    create_state_representation,
)


def outer_space_to_gym_space(space: Dict[str, Space]) -> gym.spaces.Space:
    return gym.spaces.Dict(
        {
            k: gym.spaces.Box(
                low=v.lower_bound,
                high=v.upper_bound,
                dtype=float if v.space_type is SpaceType.CONTINUOUS else int,
            )
            for k, v in space.items()
        }
    )


OuterEnvFactory = Callable[[], OuterEnv]


def from_factory(factory: OuterEnvFactory):
    return GymEnvironment(factory())


class GymEnvironment(gym.Env):
    metadata = {
        'render.modes': ['human', 'human_state', 'human_observation'],
        'video.frames_per_second': 50,
    }

    # NOTE accepting an environment instance as input is a bad idea because it
    # would need to be instantiated during gym registration
    def __init__(self, outer_env: OuterEnv):
        super().__init__()

        self.outer_env = outer_env
        self.state_space = (
            outer_space_to_gym_space(outer_env.state_representation.space)
            if outer_env.state_representation is not None
            else None
        )
        self.action_space = gym.spaces.Discrete(
            outer_env.action_space.num_actions
        )
        self.observation_space = (
            outer_space_to_gym_space(outer_env.observation_representation.space)
            if outer_env.observation_representation is not None
            else None
        )

        self._state_viewer = None
        self._observation_viewer = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        actual_seed = seeding.create_seed(seed)
        self.outer_env.inner_env.set_seed(actual_seed)
        return [actual_seed]

    def set_state_representation(self, name: str):
        """Change underlying state representation"""
        # TODO: test
        self.outer_env.state_representation = create_state_representation(
            name, self.outer_env.inner_env.state_space
        )
        self.state_space = outer_space_to_gym_space(
            self.outer_env.state_representation.space
        )

    def set_observation_representation(self, name: str):
        """Change underlying observation representation"""
        # TODO: test
        self.outer_env.observation_representation = (
            create_observation_representation(
                name, self.outer_env.inner_env.observation_space
            )
        )
        self.observation_space = outer_space_to_gym_space(
            self.outer_env.observation_representation.space
        )

    @property
    def state(self) -> Dict[str, np.ndarray]:
        return self.outer_env.state

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        return self.outer_env.observation

    def reset(self) -> Dict[str, np.ndarray]:
        """reset the environment state

        Returns:
            Dict[str, numpy.ndarray]: initial observation
        """
        self.outer_env.reset()
        return self.observation

    def step(self, action: int):
        """performs environment step

        Args:
            action (int): agent's action

        Returns:
            Tuple[Dict[str, numpy.ndarray], float, bool, Dict]: (observation, reward, terminal, info dictionary)
        """
        action_ = self.outer_env.action_space.int_to_action(action)
        reward, done = self.outer_env.step(action_)
        return self.observation, reward, done, {}

    def render(self, mode='human'):
        # TODO: test
        # only import rendering if actually rendering (avoid importing when
        # using library remotely using ssh on a display-less environment)
        from gym_gridverse.rendering import GridVerseViewer

        if mode not in ['human', 'human_state', 'human_observation']:
            super().render(mode)

        # not reset yet
        if self.outer_env.inner_env.state is None:
            return

        if mode in ['human', 'human_state']:

            if self._state_viewer is None:
                self._state_viewer = GridVerseViewer(
                    self.outer_env.inner_env.state_space.grid_shape,
                    caption='State',
                )

                # without sleep the first frame could be black
                time.sleep(0.05)

            self._state_viewer.render(self.outer_env.inner_env.state)

        if mode in ['human', 'human_observation']:

            if self._observation_viewer is None:
                self._observation_viewer = GridVerseViewer(
                    self.outer_env.inner_env.observation_space.grid_shape,
                    caption='Observation',
                )

                # without sleep the first frame could be black
                time.sleep(0.05)

            self._observation_viewer.render(
                self.outer_env.inner_env.observation
            )

    def close(self):
        # TODO: test
        if self._state_viewer is not None:
            self._state_viewer.close()
            self._state_viewer = None

        if self._observation_viewer is not None:
            self._observation_viewer.close()
            self._observation_viewer = None


class GymStateWrapper(gym.Wrapper):
    """
    Gym Wrapper to replace the standard observation representation with state instead.

    Doesn't change underlying environment, won't change render
    """

    def __init__(self, env: GymEnvironment):
        # Make sure we have a valid state representation
        if env.state_space is None:
            ValueError('GymEnvironment does not have a state space')

        super().__init__(env)
        self.observation_space = env.state_space

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        return self.env.state

    def reset(self) -> Dict[str, np.ndarray]:
        """reset the environment state

        Returns:
            Dict[str, numpy.ndarray]: initial state
        """
        self.env.reset()
        return self.observation

    def step(self, action: int):
        """performs environment step

        Args:
            action (int): agent's action

        Returns:
            Tuple[Dict[str, numpy.ndarray], float, bool, Dict]: (state, reward, terminal, info dictionary)
        """
        observation, reward, done, info = self.env.step(action)
        info['observation'] = observation
        return self.observation, reward, done, info


STRING_TO_YAML_FILE: Dict[str, str] = {
    "GV-Crossing-5x5-v0": "gv_crossing.5x5.yaml",
    "GV-Crossing-7x7-v0": "gv_crossing.7x7.yaml",
    "GV-Dynamic-obstacles-5x5-v0": "gv_dynamic_obstacles.5x5.yaml",
    "GV-Dynamic-obstacles-7x7-v0": "gv_dynamic_obstacles.7x7.yaml",
    "GV-Empty-4x4-v0": "gv_empty.4x4.yaml",
    "GV-Empty-8x8-v0": "gv_empty.8x8.yaml",
    "GV-Four-rooms-7x7-v0": "gv_four_rooms.7x7.yaml",
    "GV-Four-rooms-9x9-v0": "gv_four_rooms.9x9.yaml",
    "GV-Keydoor-5x5-v0": "gv_keydoor.5x5.yaml",
    "GV-Keydoor-7x7-v0": "gv_keydoor.7x7.yaml",
    "GV-Keydoor-9x9-v0": "gv_keydoor.9x9.yaml",
    "GV-Memory-5x5-v0": "gv_memory.5x5.yaml",
    "GV-Memory-9x9-v0": "gv_memory.9x9.yaml",
    "GV-Memory-four-rooms-7x7-v0": "gv_memory_four_rooms.7x7.yaml",
    "GV-Memory-four-rooms-9x9-v0": "gv_memory_four_rooms.9x9.yaml",
    "GV-Memory-nine-rooms-10x10-v0": "gv_memory_nine_rooms.10x10.yaml",
    "GV-Memory-nine-rooms-13x13-v0": "gv_memory_nine_rooms.13x13.yaml",
    "GV-Nine-rooms-10x10-v0": "gv_nine_rooms.10x10.yaml",
    "GV-Nine-rooms-13x13-v0": "gv_nine_rooms.13x13.yaml",
    "GV-Teleport-5x5-v0": "gv_teleport.5x5.yaml",
    "GV-Teleport-7x7-v0": "gv_teleport.7x7.yaml",
}


def outer_env_factory(yaml_filename: str) -> OuterEnv:
    env = factory_env_from_yaml(yaml_filename)
    observation_representation = create_observation_representation(
        'default', env.observation_space
    )
    return OuterEnv(
        env,
        observation_representation=observation_representation,
    )


for key, yaml_filename in STRING_TO_YAML_FILE.items():

    # registering using factory to avoid allocation of outer envs
    gym.register(
        key,
        entry_point='gym_gridverse.gym:from_factory',
        kwargs={
            'factory': partial(
                outer_env_factory, f'.registered_environments/{yaml_filename}'
            )
        },
    )

env_ids = list(STRING_TO_YAML_FILE.keys())
