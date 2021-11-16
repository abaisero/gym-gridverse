#!/usr/bin/env python
import argparse
import itertools as itt
import random
import time

from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml


def make_env(path: str) -> InnerEnv:
    """Makes a GV "inner" environment."""
    return factory_env_from_yaml(path)


def main(args):
    env = make_env(args.path)
    env.reset()

    spf = 1 / args.fps

    for ei in itt.count():
        print(f'# Episode {ei}')
        print()

        env.reset()
        print('observation:')
        print(env.observation)
        time.sleep(spf)

        for ti in itt.count():
            print(f'episode: {ei}')
            print(f'time: {ti}')

            action = random.choice(env.action_space.actions)
            reward, done = env.step(action)

            print(f'action: {action}')
            print(f'reward: {reward}')
            print('observation:')
            print(env.observation)
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
