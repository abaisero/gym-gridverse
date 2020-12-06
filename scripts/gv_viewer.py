#!/usr/bin/env python
from __future__ import annotations

import argparse
import enum
import itertools as itt
import time
from typing import Dict, Union

import pyglet

from gym_gridverse.actions import Actions
from gym_gridverse.envs import observation_functions as observation_fs
from gym_gridverse.envs.gridworld import GridWorld
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.rendering import GridVerseViewer


class Controls(enum.Enum):
    PASS = 0
    QUIT = enum.auto()
    RESET = enum.auto()
    HIDE_STATE = enum.auto()
    FLIP_HUD = enum.auto()
    CYCLE_OBSERVATION = enum.auto()


Command = Union[Actions, Controls]


class KeyboardHandler:
    keymap: Dict[int, Command] = {
        # actions
        pyglet.window.key.UP: Actions.MOVE_FORWARD,
        pyglet.window.key.DOWN: Actions.MOVE_BACKWARD,
        pyglet.window.key.LEFT: Actions.TURN_LEFT,
        pyglet.window.key.RIGHT: Actions.TURN_RIGHT,
        pyglet.window.key.W: Actions.MOVE_FORWARD,
        pyglet.window.key.A: Actions.MOVE_LEFT,
        pyglet.window.key.S: Actions.MOVE_BACKWARD,
        pyglet.window.key.D: Actions.MOVE_RIGHT,
        pyglet.window.key.SPACE: Actions.ACTUATE,
        pyglet.window.key.P: Actions.PICK_N_DROP,
        # controls
        pyglet.window.key.Q: Controls.QUIT,
        pyglet.window.key.R: Controls.RESET,
        pyglet.window.key.H: Controls.HIDE_STATE,
        pyglet.window.key.U: Controls.FLIP_HUD,
        pyglet.window.key.O: Controls.CYCLE_OBSERVATION,
    }

    def __init__(self):
        self._key = None

    def on_key_press(  # pylint: disable=unused-argument
        self, symbol, modifiers
    ):
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

    print(fstr('<UP>', Actions.MOVE_FORWARD))
    print(fstr('<DOWN>', Actions.MOVE_BACKWARD))
    print(fstr('<LEFT>', Actions.TURN_LEFT))
    print(fstr('<RIGHT>', Actions.TURN_RIGHT))
    print()

    print(fstr('w', Actions.MOVE_FORWARD))
    print(fstr('a', Actions.MOVE_LEFT))
    print(fstr('s', Actions.MOVE_BACKWARD))
    print(fstr('d', Actions.MOVE_RIGHT))
    print()

    print(fstr('<SPACE>', Actions.ACTUATE))
    print(fstr('p', Actions.PICK_N_DROP))
    print()

    print(fstr('q', Controls.QUIT))
    print(fstr('r', Controls.RESET))
    print(fstr('h', Controls.HIDE_STATE))
    print(fstr('u', Controls.FLIP_HUD))
    print(fstr('o', Controls.CYCLE_OBSERVATION))


def set_observation_function(env: GridWorld, name: str):
    """Hacks into `env` in order to set the observation function used

    XXX: uses private members and implementation knowledge. The observation
    function used in `GridWorld` is meant to be static, so there is no API
    available for this functionality. Hence this is more of a hack

    Args:
        env: grid world to set the observation function for
        name: name of observation function
    """
    observation_function = observation_fs.factory(
        name, observation_space=env.observation_space
    )

    # pylint: disable=protected-access

    # XXX: hack
    env._functional_observation = observation_function

    # ensure a new observation is generated
    env._observation = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_path', help='env YAML file')
    parser.add_argument(
        '--fps', type=float, default=30.0, help='frames per second'
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

    observation_function_names = itt.cycle(
        [
            'full_observation',
            'minigrid_observation',
            'raytracing_observation',
        ]
    )

    def make_compute_return(discount: float):
        cumreturn = 0.0
        cumdiscount = 1.0

        def compute_return(reward: float) -> float:
            nonlocal cumreturn
            nonlocal cumdiscount
            cumreturn += cumdiscount * reward
            cumdiscount *= discount
            return cumreturn

        return compute_return

    reset = True
    while True:
        if reset:
            env.reset()
            compute_return = make_compute_return(0.99)
            hud_info = {
                'action': None,
                'reward': None,
                'ret': None,
                'done': None,
            }

            state_viewer.render(env.state, **hud_info)
            observation_viewer.render(env.observation, **hud_info)
            done, reset = False, False

        command = keyboard_handler.get_command()

        if isinstance(command, Actions):
            action = command

            if not done and env.action_space.contains(action):
                reward, done = env.step(action)
                ret = compute_return(reward)

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
            continue

        elif command is Controls.QUIT:
            break

        elif command is Controls.CYCLE_OBSERVATION:
            name = next(observation_function_names)
            print(f'setting observation function: {name}')
            set_observation_function(env, name)

        state_viewer.render(env.state, **hud_info)
        observation_viewer.render(env.observation, **hud_info)
        time.sleep(args.spf)


if __name__ == '__main__':
    main()
