#!/usr/bin/env python
import argparse

import gym
import tqdm

from gym_gridverse.debugging import reset_gv_debug
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import GymEnvironment
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)


def make_env(id_or_path: str) -> GymEnvironment:
    try:
        env = gym.make(id_or_path)

    except gym.error.Error:
        inner_env = factory_env_from_yaml(id_or_path)
        state_representation = make_state_representation(
            'default',
            inner_env.state_space,
        )
        observation_representation = make_observation_representation(
            'default',
            inner_env.observation_space,
        )
        outer_env = OuterEnv(
            inner_env,
            state_representation=state_representation,
            observation_representation=observation_representation,
        )
        env = GymEnvironment(outer_env)

    else:
        if not isinstance(env, GymEnvironment):
            raise ValueError(
                f'gym id {id_or_path} is not associated with a GridVerse environment'
            )

    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id_or_path', help='Gym env id or env YAML file')
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    args = parser.parse_args()

    reset_gv_debug(False)

    env = make_env(args.id_or_path)
    env.reset()

    for _ in tqdm.trange(args.timesteps):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

        if done:
            env.reset()
