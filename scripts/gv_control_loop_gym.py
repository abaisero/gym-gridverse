#!/usr/bin/env python
import argparse
import itertools as itt
import time

import gym

from gym_gridverse.debugging import checkraise
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import GymEnvironment
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    DefaultObservationRepresentation,
)
from gym_gridverse.representations.state_representations import (
    DefaultStateRepresentation,
)


def make_env(id_or_path: str) -> GymEnvironment:
    try:
        print('Loading using gym.make')
        env = gym.make(id_or_path)

    except gym.error.Error:
        print(f'Environment with id {id_or_path} not found.')
        print('Loading using YAML')
        inner_env = factory_env_from_yaml(id_or_path)
        obs_rep = DefaultObservationRepresentation(inner_env.observation_space)
        state_rep = DefaultStateRepresentation(inner_env.state_space)
        outer_env = OuterEnv(
            inner_env, observation_rep=obs_rep, state_rep=state_rep
        )
        env = GymEnvironment.from_environment(outer_env)

    else:
        checkraise(
            lambda: isinstance(env, GymEnvironment),
            ValueError,
            'gym id {} is not associated with a GridVerse environment',
            id_or_path,
        )

    return env


def print_observation(observation):
    printable_observation = {k: v.tolist() for k, v in observation.items()}
    print('observation:')
    print(f'{printable_observation}')


def main(args):
    env = make_env(args.id_or_path)
    env.reset()

    spf = 1 / args.fps

    for ei in itt.count():
        print(f'# Episode {ei}')
        print()

        observation = env.reset()
        env.render()

        print_observation(observation)
        print()

        time.sleep(spf)

        for ti in itt.count():
            print(f'episode: {ei}')
            print(f'time: {ti}')

            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            env.render()

            print(f'action: {action}')
            print(f'reward: {reward}')
            print_observation(observation)
            print(f'done: {done}')
            print()

            time.sleep(spf)

            if done:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('id_or_path', help='Gym env id or env YAML file')
    parser.add_argument(
        '--fps', type=float, default=1.0, help='frames per second'
    )
    main(parser.parse_args())
