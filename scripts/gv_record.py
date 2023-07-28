#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools as itt
import sys
from typing import Tuple

import imageio
import numpy.random as rnd

from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.envs.observation_functions import (
    factory as observation_function_factory,
)
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.observation import Observation
from gym_gridverse.recording import Data, DataBuilder, generate_images, record
from gym_gridverse.state import State


def main():
    args = get_args()

    env = factory_env_from_yaml(args.yaml)

    if args.observation_function is not None:
        env._observation_function = observation_function_factory(
            args.observation_function, observation_space=env.observation_space
        )
        env._observation = None

    env.set_seed(args.seed)
    rnd.seed(args.seed)

    state_data, observation_data = make_data(
        env, args.discount, max_steps=args.max_steps
    )

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


def make_data(
    env: InnerEnv, discount: float, *, max_steps: int
) -> Tuple[Data[State], Data[Observation]]:
    state_data_builder: DataBuilder[State] = DataBuilder(discount)
    observation_data_builder: DataBuilder[Observation] = DataBuilder(discount)

    env.reset()

    state_data_builder.append0(env.state)
    observation_data_builder.append0(env.observation)

    for _ in range(max_steps):
        i = rnd.choice(len(env.action_space.actions))
        action = env.action_space.actions[i]
        reward, done = env.step(action)

        state_data_builder.append(env.state, action, reward)
        observation_data_builder.append(env.observation, action, reward)

        if done:
            break

    return state_data_builder.build(), observation_data_builder.build()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['images', 'gif', 'mp4'])
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
        '--max-steps', type=int, default=99, help='maximum number of steps'
    )

    parser.add_argument('--state', default=None, help='state filename')
    parser.add_argument(
        '--observation', default=None, help='observation filename'
    )

    parser.add_argument('--observation-function', default=None)

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
