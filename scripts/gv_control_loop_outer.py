#!/usr/bin/env python
import argparse
import itertools as itt
import random
import time
from typing import Dict

import numpy as np

from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)


def make_env(path: str) -> OuterEnv:
    """Makes a GV "outer" environment."""
    inner_env = factory_env_from_yaml(path)
    state_representation = make_state_representation(
        'default',
        inner_env.state_space,
    )
    observation_representation = make_observation_representation(
        'default',
        inner_env.observation_space,
    )
    return OuterEnv(
        inner_env,
        state_representation=state_representation,
        observation_representation=observation_representation,
    )


def print_compact(data: Dict[str, np.ndarray]):
    """Converts numpy arrays into lists before printing, for more compact output."""
    compact_data = {k: v.tolist() for k, v in data.items()}
    print(compact_data)


def main(args):
    env = make_env(args.path)
    env.reset()

    spf = 1 / args.fps

    for ei in itt.count():
        print(f'# Episode {ei}')
        print()

        env.reset()
        print('state:')
        print_compact(env.state)
        print('observation:')
        print_compact(env.observation)
        time.sleep(spf)

        for ti in itt.count():
            print(f'episode: {ei}')
            print(f'time: {ti}')

            action = random.choice(env.action_space.actions)
            reward, done = env.step(action)

            print(f'action: {action}')
            print(f'reward: {reward}')
            print('state:')
            print_compact(env.state)
            print('observation:')
            print_compact(env.observation)
            print(f'done: {done}')
            print()

            time.sleep(spf)

            if done:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='env YAML file')
    parser.add_argument(
        '--fps', type=float, default=1.0, help='frames per second'
    )
    main(parser.parse_args())
