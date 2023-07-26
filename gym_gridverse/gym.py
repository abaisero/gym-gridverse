from __future__ import annotations

import time
from functools import partial
from typing import Callable, Dict, List, Optional

import gym
import numpy as np
import pkg_resources
from gym.utils import seeding

from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.spaces import Space, SpaceType
from gym_gridverse.representations.state_representations import (
    make_state_representation,
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
        """Environment state space, if any."""

        self.action_space = gym.spaces.Discrete(
            outer_env.action_space.num_actions
        )
        """Environment action space."""

        self.observation_space = (
            outer_space_to_gym_space(outer_env.observation_representation.space)
            if outer_env.observation_representation is not None
            else None
        )
        """Environment observation space, if any."""

        self._state_viewer = None
        self._observation_viewer = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        actual_seed = seeding.create_seed(seed)
        self.outer_env.inner_env.set_seed(actual_seed)
        return [actual_seed]

    def set_state_representation(self, name: str):
        """Changes the state representation."""
        # TODO: test
        self.outer_env.state_representation = make_state_representation(
            name, self.outer_env.inner_env.state_space
        )
        self.state_space = outer_space_to_gym_space(
            self.outer_env.state_representation.space
        )

    def set_observation_representation(self, name: str):
        """Changes the observation representation."""
        # TODO: test
        self.outer_env.observation_representation = (
            make_observation_representation(
                name, self.outer_env.inner_env.observation_space
            )
        )
        self.observation_space = outer_space_to_gym_space(
            self.outer_env.observation_representation.space
        )

    @property
    def state(self) -> Dict[str, np.ndarray]:
        """Returns the representation of the current state."""
        return self.outer_env.state

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        """Returns the representation of the current observation."""
        return self.outer_env.observation

    def reset(self) -> Dict[str, np.ndarray]:
        """Resets the state of the environment.

        Returns:
            Dict[str, numpy.ndarray]: initial observation
        """
        self.outer_env.reset()
        return self.observation

    def step(self, action: int):
        """Runs the environment dynamics for one timestep.

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

        if mode not in [
            'human',
            'human_state',
            'human_observation',
            'rgb_array',
            'rgb_array_state',
            'rgb_array_observation',
        ]:
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

        rgb_arrays = []

        if mode in ['rgb_array', 'rgb_array_state']:
            if self._state_viewer is None:
                self._state_viewer = GridVerseViewer(
                    self.outer_env.inner_env.state_space.grid_shape,
                    caption='State',
                )

                # without sleep the first frame could be black
                time.sleep(0.05)

            rgb_array_state = self._state_viewer.render(
                self.outer_env.inner_env.state,
                return_rgb_array=True,
            )
            rgb_arrays.append(rgb_array_state)

        if mode in ['rgb_array', 'rgb_array_observation']:
            if self._observation_viewer is None:
                self._observation_viewer = GridVerseViewer(
                    self.outer_env.inner_env.observation_space.grid_shape,
                    caption='Observation',
                )

                # without sleep the first frame could be black
                time.sleep(0.05)

            rgb_array_observation = self._observation_viewer.render(
                self.outer_env.inner_env.observation,
                return_rgb_array=True,
            )
            rgb_arrays.append(rgb_array_observation)

        if rgb_arrays:
            return tuple(rgb_arrays) if len(rgb_arrays) > 1 else rgb_arrays[0]

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
    "GV-DynamicObstacles-5x5-v0": "gv_dynamic_obstacles.5x5.yaml",
    "GV-DynamicObstacles-7x7-v0": "gv_dynamic_obstacles.7x7.yaml",
    "GV-Empty-4x4-v0": "gv_empty.4x4.yaml",
    "GV-Empty-8x8-v0": "gv_empty.8x8.yaml",
    "GV-FourRooms-7x7-v0": "gv_four_rooms.7x7.yaml",
    "GV-FourRooms-9x9-v0": "gv_four_rooms.9x9.yaml",
    "GV-Keydoor-5x5-v0": "gv_keydoor.5x5.yaml",
    "GV-Keydoor-7x7-v0": "gv_keydoor.7x7.yaml",
    "GV-Keydoor-9x9-v0": "gv_keydoor.9x9.yaml",
    "GV-Memory-5x5-v0": "gv_memory.5x5.yaml",
    "GV-Memory-9x9-v0": "gv_memory.9x9.yaml",
    "GV-MemoryFourRooms-7x7-v0": "gv_memory_four_rooms.7x7.yaml",
    "GV-MemoryFourRooms-9x9-v0": "gv_memory_four_rooms.9x9.yaml",
    "GV-MemoryNineRooms-10x10-v0": "gv_memory_nine_rooms.10x10.yaml",
    "GV-MemoryNineRooms-13x13-v0": "gv_memory_nine_rooms.13x13.yaml",
    "GV-NineRooms-10x10-v0": "gv_nine_rooms.10x10.yaml",
    "GV-NineRooms-13x13-v0": "gv_nine_rooms.13x13.yaml",
    "GV-Teleport-5x5-v0": "gv_teleport.5x5.yaml",
    "GV-Teleport-7x7-v0": "gv_teleport.7x7.yaml",
}


def outer_env_factory(yaml_filename: str) -> OuterEnv:
    env = factory_env_from_yaml(yaml_filename)
    observation_representation = make_observation_representation(
        'default', env.observation_space
    )
    return OuterEnv(
        env,
        observation_representation=observation_representation,
    )


for key, yaml_filename in STRING_TO_YAML_FILE.items():
    yaml_filepath = pkg_resources.resource_filename(
        'gym_gridverse', f'registered_envs/{yaml_filename}'
    )
    factory = partial(outer_env_factory, yaml_filepath)

    # registering using factory to avoid allocation of outer envs
    gym.register(
        key,
        entry_point='gym_gridverse.gym:from_factory',
        kwargs={'factory': factory},
    )

env_ids = list(STRING_TO_YAML_FILE.keys())
