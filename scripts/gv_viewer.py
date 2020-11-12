#!/usr/bin/env python
from __future__ import annotations

import argparse
import enum
import math
import random
import time
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import pyglet
from gym.envs.classic_control import rendering

from gym_gridverse.actions import Actions
from gym_gridverse.envs import Environment
from gym_gridverse.envs.factory_yaml import make_environment
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Floor,
    Goal,
    Key,
    MovingObstacle,
    Wall,
)
from gym_gridverse.spaces import StateSpace
from gym_gridverse.state import State


class Group(rendering.Geom):
    """like rendering.Compound, but without sharing attributes"""

    def __init__(self, geoms: Sequence[rendering.Geom]):
        super().__init__()
        self.geoms = geoms

    def render1(self):
        for geom in self.geoms:
            geom.render()


def make_grid(  # pylint: disable=too-many-locals
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_rows: int,
    num_cols: int,
) -> rendering.Geom:

    start_x, start_y = start
    end_x, end_y = end

    lines = []

    for i in range(num_rows + 1):
        t = i / num_rows
        line_y = (1.0 - t) * start_y + t * end_y
        line_start = start_x, line_y
        line_end = end_x, line_y
        line = rendering.Line(line_start, line_end)
        lines.append(line)

    for j in range(num_cols + 1):
        t = j / num_cols
        line_x = (1.0 - t) * start_x + t * end_x
        line_start = line_x, start_y
        line_end = line_x, end_y
        line = rendering.Line(line_start, line_end)
        lines.append(line)

    geom = Group(lines)
    return geom


def make_grid_background() -> rendering.Geom:
    geom = rendering.make_polygon(
        [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0),], filled=True,
    )
    geom.set_color(0.65, 0.65, 0.65)
    return geom


# brick red
NONE = (0.5, 0.5, 0.5)
RED = (0.796, 0.255, 0.329)
GREEN = (0.329, 0.796, 0.255)
BLUE = (0.255, 0.329, 0.796)
YELLOW = (0.796, 0.796, 0.329)


colormap = {
    Colors.NONE: NONE,
    Colors.RED: RED,
    Colors.GREEN: GREEN,
    Colors.BLUE: BLUE,
    Colors.YELLOW: YELLOW,
}


def make_agent() -> rendering.Geom:
    pad = 0.7
    geom_agent = rendering.make_polygon(
        [(-pad, -pad), (0.0, pad), (pad, -pad)], filled=False
    )
    geom_agent.set_linewidth(3)
    geom_agent.set_color(*BLUE)
    return geom_agent


def make_goal(goal: Goal) -> rendering.Geom:  # pylint: disable=unused-argument
    geom_goal = rendering.make_polygon(
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)], filled=True,
    )
    geom_goal.set_color(*GREEN)

    pad = 0.8
    geom_flag = rendering.make_polyline(
        [(0.0, -pad), (0.0, pad), (pad, pad / 2), (0.0, 0.0)]
    )
    geom_flag.set_linewidth(2)
    geom_flag.add_attr(rendering.Transform(translation=(-pad / 4, 0.0)))

    return Group([geom_goal, geom_flag])


def make_wall(wall: Wall) -> rendering.Geom:  # pylint: disable=unused-argument
    geom_background = rendering.make_polygon(
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)], filled=True,
    )
    geom_background.set_color(*RED)

    geom_tile_lines = [
        # horizontal
        rendering.Line((-1.0, -0.33), (1.0, -0.33)),
        rendering.Line((-1.0, 0.33), (1.0, 0.33)),
        # vertical
        rendering.Line((-0.5, -1.0), (-0.5, -0.33)),
        rendering.Line((0.5, -1.0), (0.5, -0.33)),
        # vertical
        rendering.Line((0.0, -0.33), (0.0, 0.33)),
        # vertical
        rendering.Line((-0.5, 1.0), (-0.5, 0.33)),
        rendering.Line((0.5, 1.0), (0.5, 0.33)),
    ]

    return Group([geom_background, *geom_tile_lines])


def _make_door_open(  # pylint: disable=unused-argument
    door: Door,
) -> rendering.Geom:
    pad = 0.8

    geoms_frame_background = [
        rendering.make_polygon(
            [(-1.0, -1.0), (-1.0, 1.0), (-pad, 1.0), (-pad, -1.0)], filled=True,
        ),
        rendering.make_polygon(
            [(pad, -1.0), (pad, 1.0), (1.0, 1.0), (1.0, -1.0)], filled=True,
        ),
        rendering.make_polygon(
            [(-1.0, -1.0), (-1.0, -pad), (1.0, -pad), (1.0, -1.0)], filled=True,
        ),
        rendering.make_polygon(
            [(-1.0, pad), (-1.0, 1.0), (1.0, 1.0), (1.0, pad)], filled=True,
        ),
    ]
    geom_frame_background = rendering.Compound(geoms_frame_background)
    geom_frame_background.set_color(*colormap[door.color])

    geom_frame = rendering.make_polygon(
        [(-pad, -pad), (-pad, pad), (pad, pad), (pad, -pad)], filled=False,
    )

    return Group([geom_frame_background, geom_frame])


def _make_door_closed_locked(  # pylint: disable=unused-argument
    door: Door,
) -> rendering.Geom:

    geom_door = rendering.make_polygon(
        # [(-0.8, -0.8), (-0.8, 0.8), (0.8, 0.8), (0.8, -0.8)], filled=True,
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)],
        filled=True,
    )
    geom_door.set_color(*colormap[door.color])

    pad = 0.8
    geom_frame = rendering.make_polygon(
        [(-pad, -pad), (-pad, pad), (pad, pad), (pad, -pad)], filled=False,
    )

    geom_keyhole = Group(
        [
            rendering.make_circle(0.2, 10, filled=True),
            rendering.make_polygon(
                [(-0.2, -0.4), (0.0, 0.0), (0.2, -0.4)], filled=True
            ),
        ]
    )
    geom_keyhole.add_attr(rendering.Transform(translation=(0.4, 0.0)))

    return Group([geom_door, geom_frame, geom_keyhole])


def _make_door_closed_unlocked(  # pylint: disable=unused-argument
    door: Door,
) -> rendering.Geom:

    geom_door = rendering.make_polygon(
        # [(-0.8, -0.8), (-0.8, 0.8), (0.8, 0.8), (0.8, -0.8)], filled=True,
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)],
        filled=True,
    )
    geom_door.set_color(*colormap[door.color])

    pad = 0.8
    geom_frame = rendering.make_polygon(
        [(-pad, -pad), (-pad, pad), (pad, pad), (pad, -pad)], filled=False,
    )

    geom_handle = rendering.make_circle(radius=0.2, res=10, filled=False)
    geom_handle.add_attr(rendering.Transform(translation=(0.4, 0.0)))

    return Group([geom_door, geom_frame, geom_handle])


def make_door(door: Door) -> rendering.Geom:
    return (
        _make_door_open(door)
        if door.is_open
        else _make_door_closed_locked(door)
        if door.locked
        else _make_door_closed_unlocked(door)
    )


def make_key(key: Key) -> rendering.Geom:  # pylint: disable=unused-argument
    geom_bow = rendering.make_circle(radius=0.4, res=6, filled=True)
    geom_bow.add_attr(rendering.Transform(translation=(-0.3, 0.0)))

    geom_blade = rendering.make_capsule(0.6, 0.2)

    geom_bit1 = rendering.make_capsule(0.3, 0.1)
    geom_bit1.add_attr(
        rendering.Transform(translation=(0.4, 0.0), rotation=math.pi / 2)
    )

    geom_bit2 = rendering.make_capsule(0.2, 0.1)
    geom_bit2.add_attr(
        rendering.Transform(translation=(0.5, 0.0), rotation=math.pi / 2)
    )

    geom_bit3 = rendering.make_capsule(0.3, 0.1)
    geom_bit3.add_attr(
        rendering.Transform(translation=(0.6, 0.0), rotation=math.pi / 2)
    )

    geom = rendering.Compound(
        [geom_bow, geom_blade, geom_bit1, geom_bit2, geom_bit3]
    )
    geom.set_color(*colormap[key.color])
    return geom


def make_moving_obstacle(  # pylint: disable=unused-argument
    obstacle: MovingObstacle,
) -> rendering.Geom:

    pad = 0.8
    geom = rendering.make_polygon(
        [(-pad, 0.0), (0.0, pad), (pad, 0.0), (0.0, -pad)], filled=False
    )
    return geom


def convert_pos(position: Position, *, num_rows: int,) -> Tuple[float, float]:
    return 2 * position.x, 2 * (num_rows - 1 - position.y)


class GridVerseViewer:
    def __init__(self, state_space: StateSpace):
        self.state_space = state_space

        self._pos_converter = partial(
            convert_pos, num_rows=state_space.grid_shape.height,
        )

        self.viewer_transforms = [
            rendering.Transform(translation=(1.0, 1.0)),
            rendering.Transform(
                scale=(
                    0.5 / state_space.grid_shape.width,
                    0.5 / state_space.grid_shape.height,
                ),
            ),
        ]

        self._viewer = rendering.Viewer(500, 500)
        self._viewer.set_bounds(0.0, 1.0, 0.0, 1.0)

        background = make_grid_background()
        self._viewer.add_geom(background)

        self.grid = make_grid(
            (0.0, 0.0),
            (1.0, 1.0),
            state_space.grid_shape.height,
            state_space.grid_shape.width,
        )

    def __del__(self):
        self._viewer.close()

    @property
    def window(self) -> pyglet.window.Window:
        return self._viewer.window

    @classmethod
    def from_env(cls, env: Environment) -> GridVerseViewer:
        return cls(env.state_space)

    def render_state(self, state: State):
        for position in state.grid.positions():
            obj = state.grid[position]
            if isinstance(obj, Floor):
                pass

            if isinstance(obj, Wall):
                geom = make_wall(obj)
                self.draw_geom_onetime(geom, position)

            if isinstance(obj, Key):
                geom = make_key(obj)
                self.draw_geom_onetime(geom, position)

            if isinstance(obj, Door):
                geom = make_door(obj)
                self.draw_geom_onetime(geom, position)

            if isinstance(obj, Goal):
                geom = make_goal(obj)
                self.draw_geom_onetime(geom, position)

            if isinstance(obj, MovingObstacle):
                geom = make_moving_obstacle(obj)
                self.draw_geom_onetime(geom, position)

        geom = make_agent()
        self.draw_geom_onetime(
            geom, state.agent.position, state.agent.orientation
        )

        self._viewer.add_onetime(self.grid)
        self._viewer.render(return_rgb_array=True)

    def draw_geom_onetime(
        self,
        geom: rendering.Geom,
        position: Position,
        orientation: Orientation = Orientation.N,
    ):
        geom.add_attr(rendering.Transform(rotation=orientation.as_radians()))
        geom.add_attr(
            rendering.Transform(translation=self._pos_converter(position))
        )
        for transform in self.viewer_transforms:
            geom.add_attr(transform)
        self._viewer.add_onetime(geom)


class Control(enum.Enum):
    PASS = 0
    QUIT = enum.auto()
    RESET = enum.auto()


Command = Union[Actions, Control]


def main_random_control(env, viewer, args):
    done = True
    while True:
        if done:
            env.reset()
            viewer.render_state(env.state)
            done = False

        action = random.choice(env.action_space.actions)
        _, done = env.step(action)

        viewer.render_state(env.state)
        time.sleep(args.spf)


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
        pyglet.window.key.R: Control.RESET,
        pyglet.window.key.Q: Control.QUIT,
    }

    def __init__(self):
        self.key = None

    def on_key_press(  # pylint: disable=unused-argument
        self, symbol, modifiers
    ):
        self.key = symbol

    def get_command(self) -> Command:
        try:
            return self.keymap[self.key]
        except KeyError:
            return Control.PASS
        finally:
            self.key = None


def main_manual_control(env, viewer, args):
    keyboard_handler = KeyboardHandler()
    viewer.window.push_handlers(keyboard_handler)

    done = True
    while True:
        if done:
            env.reset()
            viewer.render_state(env.state)
            done = False

        command = keyboard_handler.get_command()

        if isinstance(command, Actions):
            _, done = env.step(command)

        elif command is Control.RESET:
            raise NotImplementedError

        elif command is Control.QUIT:
            raise NotImplementedError

        viewer.render_state(env.state)
        time.sleep(args.spf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='YAML data file')
    parser.add_argument(
        '--fps', type=float, default=15.0, help='frames per second'
    )
    parser.add_argument(
        '--manual-control', action='store_true', help='Manual control'
    )
    args = parser.parse_args()
    args.spf = 1 / args.fps

    with open(args.data_path) as f:
        env = make_environment(f)

    viewer = GridVerseViewer.from_env(env)

    if args.manual_control:
        main_manual_control(env, viewer, args)
    else:
        main_random_control(env, viewer, args)


if __name__ == '__main__':
    main()
