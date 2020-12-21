#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools as itt
import sys
from typing import List, Tuple

import imageio
import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.observation import Observation
from gym_gridverse.recording import Data, generate_images, record
from gym_gridverse.state import State


def main():
    args = get_args()

    env = factory_env_from_yaml(args.yaml)
    env.set_seed(args.seed)
    rnd.seed(args.seed)

    state_data, observation_data = make_data(env, args.discount)

    if args.state:
        images = list(generate_images(state_data))
        filename = args.state
        filenames = map(args.state.format, itt.count())

        record(
            args.mode,
            images,
            filename=filename,
            filenames=filenames,
            loop=args.gif_loop,
            duration=args.gif_duration,
            fps=args.gif_fps,
        )

    if args.observation:
        images = list(generate_images(observation_data))
        filename = args.observation
        filenames = map(args.observation.format, itt.count())

        record(
            args.mode,
            images,
            filename=filename,
            filenames=filenames,
            loop=args.gif_loop,
            duration=args.gif_duration,
            fps=args.gif_fps,
        )


def make_data(env: InnerEnv, discount: float) -> Tuple[Data, Data]:
    states: List[State] = []
    observations: List[Observation] = []
    actions: List[Action] = []
    rewards: List[float] = []

    env.reset()

    states.append(env.state)
    observations.append(env.observation)

    done = False
    while not done:
        action = rnd.choice(env.action_space.actions)
        reward, done = env.step(action)

        states.append(env.state)
        observations.append(env.observation)
        actions.append(action)
        rewards.append(reward)

    return (
        Data(states, actions, rewards, discount),
        Data(observations, actions, rewards, discount),
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['images', 'gif'])
    parser.add_argument('yaml', help='env YAML file')

    parser.add_argument('--seed', type=int, default=None, help='env seed')

    parser.add_argument(
        '--gif-loop', type=int, default=0, help='gif loop count'
    )
    parser.add_argument(
        '--gif-duration', type=float, default=None, help='gif duration'
    )
    parser.add_argument('--gif-fps', type=float, default=2.0, help='gif fps')

    parser.add_argument(
        '--discount', type=float, default=1.0, help='discount factor'
    )
    parser.add_argument(
        '--max-steps', type=int, default=100, help='maximum number of steps'
    )

    parser.add_argument('--state', default=None, help='state filename')
    parser.add_argument(
        '--observation', default=None, help='observation filename'
    )

    imageio_help_sentinel = object()  # used to detect no argument given
    parser.add_argument(
        '--imageio-help',
        nargs='?',
        const=imageio_help_sentinel,
        help='run imageio.help(name) and exit',
    )

    args = parser.parse_args()

    if args.imageio_help is not None:
        name = (
            args.imageio_help
            if args.imageio_help is not imageio_help_sentinel
            else None
        )
        imageio.help(name)
        sys.exit(0)

    if args.state is None and args.observation is None:
        raise ValueError(
            'you must give at least --state or --observation (or both)'
        )

    return args


if __name__ == '__main__':
    main()
