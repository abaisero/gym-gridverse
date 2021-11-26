#!/usr/bin/env python
from __future__ import annotations

import argparse
import enum
import itertools as itt
import time
from typing import Dict, Union

import numpy as np
import pyglet

from gym_gridverse.action import Action
from gym_gridverse.envs.gridworld import GridWorld
from gym_gridverse.envs.observation_functions import (
    ObservationFunction,
    factory as observation_function_factory,
)
from gym_gridverse.envs.visibility_functions import (
    factory as visibility_function_factory,
    visibility_function_registry,
)
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.recording import (
    DataBuilder,
    HUD_Info,
    generate_images,
    record_gif,
)
from gym_gridverse.rendering import GridVerseViewer
from gym_gridverse.utils.rl import make_return_computer


class Controls(enum.Enum):
    PASS = 0
    QUIT = enum.auto()
    RESET = enum.auto()
    HIDE_STATE = enum.auto()
    FLIP_HUD = enum.auto()
    CYCLE_OBSERVATION = enum.auto()
    RECORD_GIF = enum.auto()


Command = Union[Action, Controls]


class KeyboardHandler:
    keymap: Dict[int, Command] = {
        # actions
        pyglet.window.key.UP: Action.MOVE_FORWARD,
        pyglet.window.key.DOWN: Action.MOVE_BACKWARD,
        pyglet.window.key.LEFT: Action.TURN_LEFT,
        pyglet.window.key.RIGHT: Action.TURN_RIGHT,
        pyglet.window.key.W: Action.MOVE_FORWARD,
        pyglet.window.key.A: Action.MOVE_LEFT,
        pyglet.window.key.S: Action.MOVE_BACKWARD,
        pyglet.window.key.D: Action.MOVE_RIGHT,
        pyglet.window.key.SPACE: Action.ACTUATE,
        pyglet.window.key.P: Action.PICK_N_DROP,
        # controls
        pyglet.window.key.Q: Controls.QUIT,
        pyglet.window.key.R: Controls.RESET,
        pyglet.window.key.H: Controls.HIDE_STATE,
        pyglet.window.key.U: Controls.FLIP_HUD,
        pyglet.window.key.O: Controls.CYCLE_OBSERVATION,
        pyglet.window.key.G: Controls.RECORD_GIF,
    }

    def __init__(self):
        self._key = None

    def on_key_press(self, symbol, modifiers):
        self._key = symbol

    def get_command(self) -> Command:
        try:
            return self.keymap[self._key]
        except KeyError:
            return Controls.PASS
        finally:
            self._key = None


def print_legend():
    def fstr(label: str, e: enum.Enum):
        return f'{label:>8s} : {e.name}'

    print('LEGEND')
    print('------')

    print(fstr('<UP>', Action.MOVE_FORWARD))
    print(fstr('<DOWN>', Action.MOVE_BACKWARD))
    print(fstr('<LEFT>', Action.TURN_LEFT))
    print(fstr('<RIGHT>', Action.TURN_RIGHT))
    print()

    print(fstr('w', Action.MOVE_FORWARD))
    print(fstr('a', Action.MOVE_LEFT))
    print(fstr('s', Action.MOVE_BACKWARD))
    print(fstr('d', Action.MOVE_RIGHT))
    print()

    print(fstr('<SPACE>', Action.ACTUATE))
    print(fstr('p', Action.PICK_N_DROP))
    print()

    print(fstr('q', Controls.QUIT))
    print(fstr('r', Controls.RESET))
    print(fstr('h', Controls.HIDE_STATE))
    print(fstr('u', Controls.FLIP_HUD))
    print(fstr('o', Controls.CYCLE_OBSERVATION))
    print(fstr('g', Controls.RECORD_GIF))


def set_observation_function(
    env: GridWorld, observation_function: ObservationFunction
):
    """Hacks into `env` in order to set the observation function used

    XXX: uses private members and implementation knowledge. The observation
    function used in `GridWorld` is meant to be static, so there is no API
    available for this functionality. Hence this is more of a hack

    Args:
        env: grid world to set the observation function for
        observation_function: observation function
    """

    # XXX: hack
    env._observation_function = observation_function

    # ensure a new observation is generated
    env._observation = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_path', help='env YAML file')
    parser.add_argument(
        '--discount', type=float, default=1.0, help='discount factor'
    )
    parser.add_argument(
        '--fps', type=float, default=30.0, help='frames per second'
    )

    parser.add_argument('--gif-record', action='store_true')
    parser.add_argument(
        '--gif-filename-state',
        default='recordings/state.{}.gif',
        help='state gif filename format',
    )
    parser.add_argument(
        '--gif-filename-observation',
        default='recordings/observation.{}.gif',
        help='observation gif filename format',
    )
    parser.add_argument(
        '--gif-loop', type=int, default=0, help='gif loop count'
    )
    parser.add_argument('--gif-fps', type=float, default=2.0, help='gif fps')
    parser.add_argument(
        '--gif-duration', type=float, default=None, help='gif duration'
    )

    args = parser.parse_args()
    args.spf = 1 / args.fps

    env = factory_env_from_yaml(args.env_path)

    print_legend()

    state_viewer = GridVerseViewer(env.state_space.grid_shape, caption='State')
    observation_viewer = GridVerseViewer(
        env.observation_space.grid_shape, caption='Observation'
    )

    keyboard_handler = KeyboardHandler()
    state_viewer.window.push_handlers(keyboard_handler)
    observation_viewer.window.push_handlers(keyboard_handler)

    visibility_function_names = itt.cycle(visibility_function_registry.keys())

    state_data_builder: DataBuilder[np.ndarray]
    observation_data_builder: DataBuilder[np.ndarray]

    quit_ = False
    for episode in itt.count():
        state_data_builder = DataBuilder(args.discount)
        observation_data_builder = DataBuilder(args.discount)

        return_computer = make_return_computer(args.discount)

        hud_info: HUD_Info = {
            'action': None,
            'reward': None,
            'ret': None,
            'done': None,
        }

        env.reset()

        state_image = state_viewer.render(
            env.state, return_rgb_array=True, **hud_info
        )
        observation_image = observation_viewer.render(
            env.observation, return_rgb_array=True, **hud_info
        )

        state_data_builder.append0(state_image)
        observation_data_builder.append0(observation_image)

        done, reset = False, False
        record = True  # avoids recording the same episode multiple times
        for _ in itt.count():
            command = keyboard_handler.get_command()

            if isinstance(command, Action):
                action = command

                if not done and env.action_space.contains(action):
                    reward, done = env.step(action)
                    ret = return_computer(reward)

                    hud_info = {
                        'action': action,
                        'reward': reward,
                        'ret': ret,
                        'done': done,
                    }

            if command is Controls.HIDE_STATE:
                state_viewer.flip_visibility()

            if command is Controls.FLIP_HUD:
                state_viewer.flip_hud()
                observation_viewer.flip_hud()

            elif command is Controls.RESET:
                reset = True

            elif command is Controls.QUIT:
                quit_ = True

            elif command is Controls.CYCLE_OBSERVATION:
                name = next(visibility_function_names)
                print(f'setting observation function: {name}')
                visibility_function = visibility_function_factory(name)
                observation_function = observation_function_factory(
                    'from_visibility',
                    visibility_function=visibility_function,
                    area=env.observation_space.area,
                )
                set_observation_function(env, observation_function)

            state_image = state_viewer.render(
                env.state, return_rgb_array=True, **hud_info
            )
            observation_image = observation_viewer.render(
                env.observation, return_rgb_array=True, **hud_info
            )

            if isinstance(command, Action):
                state_data_builder.append(state_image, action, reward)
                observation_data_builder.append(
                    observation_image, action, reward
                )

            if args.gif_record and done and record:
                state_data = state_data_builder.build()
                state_images = list(generate_images(state_data))
                record_gif(
                    args.gif_filename_state.format(episode),
                    state_images,
                    loop=args.gif_loop,
                    duration=args.gif_duration,
                    fps=args.gif_fps,
                )

                observation_data = observation_data_builder.build()
                observation_images = list(generate_images(observation_data))
                record_gif(
                    args.gif_filename_observation.format(episode),
                    observation_images,
                    loop=args.gif_loop,
                    duration=args.gif_duration,
                    fps=args.gif_fps,
                )

                record = False

            time.sleep(args.spf)

            if reset or quit_:
                break

        if quit_:
            break


if __name__ == '__main__':
    main()
