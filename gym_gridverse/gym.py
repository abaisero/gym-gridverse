import time
from functools import partial
from typing import Callable, Dict, List, Optional

import gym
import numpy as np
from gym.utils import seeding

from gym_gridverse.envs import InnerEnv, factory
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.rendering import GridVerseViewer
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
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 50,
    }

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

        self._state_viewer: Optional[GridVerseViewer] = None
        self._observation_viewer: Optional[GridVerseViewer] = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        actual_seed = seeding.create_seed(seed)
        self.outer_env.inner_env.set_seed(actual_seed)
        return [actual_seed]

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
    def state(self) -> Dict[str, np.ndarray]:
        return self.outer_env.state

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        return self.outer_env.observation

    def reset(self) -> Dict[str, np.ndarray]:
        self.outer_env.reset()
        return self.observation

    def step(self, action: int):
        action_ = self.outer_env.action_space.int_to_action(action)
        reward, done = self.outer_env.step(action_)
        return self.observation, reward, done, {}

    def render(  # pylint: disable=arguments-differ
        self, mode='human', *, what='both'
    ):
        if mode not in ['human', 'rgb_array']:
            super().render(mode)

        if what not in ['state', 'observation', 'both']:
            raise ValueError(f'invalid {what}')

        # not reset yet
        if self.outer_env.inner_env.state is None:
            return

        if mode == 'human':

            if what in ['state', 'both']:

                if self._state_viewer is None:
                    self._state_viewer = GridVerseViewer(
                        self.outer_env.inner_env.state_space.grid_shape,
                        caption='State',
                    )

                    # without sleep the first frame could be black
                    time.sleep(0.05)

                self._state_viewer.render(self.outer_env.inner_env.state)

            if what in ['observation', 'both']:

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
        if self._state_viewer is not None:
            self._state_viewer.close()
            self._state_viewer = None

        if self._observation_viewer is not None:
            self._observation_viewer.close()
            self._observation_viewer = None


env_ids = []

for key, constructor_ in factory.STRING_TO_GYM_CONSTRUCTOR.items():

    def outer_env_constructor(constructor: Callable[[], InnerEnv]) -> OuterEnv:
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
