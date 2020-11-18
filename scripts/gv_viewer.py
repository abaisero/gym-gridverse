#!/usr/bin/env python
from __future__ import annotations

import argparse
import enum
import itertools as itt
import math
import time
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import pyglet
from gym.envs.classic_control import rendering

from gym_gridverse.actions import Actions
from gym_gridverse.envs import observation_functions as observation_fs
from gym_gridverse.envs.factory_yaml import make_environment
from gym_gridverse.geometry import Orientation, Position, Shape
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Floor,
    Goal,
    Hidden,
    Key,
    MovingObstacle,
    Wall,
)
from gym_gridverse.observation import Observation
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


def make_hidden(  # pylint: disable=unused-argument
    hidden: Hidden,
) -> rendering.Geom:

    geom = rendering.make_polygon(
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)], filled=True,
    )
    return geom


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


def make_capsule(length, width, *, filled=True):
    l, r, t, b = 0, length, width / 2, -width / 2
    box = rendering.make_polygon(
        [(l, b), (l, t), (r, t), (r, b)], filled=filled
    )
    circ0 = rendering.make_circle(width / 2, filled=filled)
    circ1 = rendering.make_circle(width / 2, filled=filled)
    circ1.add_attr(rendering.Transform(translation=(length, 0)))
    geom = Group([box, circ0, circ1])
    return geom


def make_key(key: Key) -> rendering.Geom:  # pylint: disable=unused-argument
    # OUTLINE

    lw = 4

    geom_bow_outline = rendering.make_circle(radius=0.4, res=6, filled=False)
    geom_bow_outline.add_attr(rendering.Transform(translation=(-0.3, 0.0)))
    geom_bow_outline.set_linewidth(lw)

    geom_blade_outline = make_capsule(0.6, 0.2, filled=False)
    for geom in geom_blade_outline.geoms:
        geom.set_linewidth(lw)

    geom_bit1_outline = make_capsule(0.3, 0.1, filled=False)
    geom_bit1_outline.add_attr(
        rendering.Transform(translation=(0.4, 0.0), rotation=math.pi / 2)
    )
    for geom in geom_bit1_outline.geoms:
        geom.set_linewidth(lw)

    geom_bit2_outline = make_capsule(0.3, 0.1, filled=False)
    geom_bit2_outline.add_attr(
        rendering.Transform(translation=(0.5, 0.0), rotation=math.pi / 2)
    )
    for geom in geom_bit2_outline.geoms:
        geom.set_linewidth(lw)

    geom_bit3_outline = make_capsule(0.3, 0.1, filled=False)
    geom_bit3_outline.add_attr(
        rendering.Transform(translation=(0.6, 0.0), rotation=math.pi / 2)
    )
    for geom in geom_bit3_outline.geoms:
        geom.set_linewidth(lw)

    geom_outline = Group(
        [
            geom_bow_outline,
            geom_blade_outline,
            geom_bit1_outline,
            geom_bit2_outline,
            geom_bit3_outline,
        ]
    )

    # BODY

    geom_bow = rendering.make_circle(radius=0.4, res=6, filled=True)
    geom_bow.add_attr(rendering.Transform(translation=(-0.3, 0.0)))

    geom_blade = rendering.make_capsule(0.6, 0.2)

    geom_bit1 = rendering.make_capsule(0.3, 0.1)
    geom_bit1.add_attr(
        rendering.Transform(translation=(0.4, 0.0), rotation=math.pi / 2)
    )

    geom_bit2 = rendering.make_capsule(0.3, 0.1)
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

    return Group([geom_outline, geom])


def make_moving_obstacle(  # pylint: disable=unused-argument
    obstacle: MovingObstacle,
) -> rendering.Geom:

    pad = 0.8
    geom = rendering.make_polygon(
        [(-pad, 0.0), (0.0, pad), (pad, 0.0), (0.0, -pad),], filled=True,
    )
    geom.set_color(*RED)

    geom_outline = rendering.make_polygon(
        [(-pad, 0.0), (0.0, pad), (pad, 0.0), (0.0, -pad),], filled=False,
    )
    geom_outline.set_linewidth(3)

    return Group([geom, geom_outline])


def convert_pos(position: Position, *, num_rows: int,) -> Tuple[float, float]:
    return 2 * position.x, 2 * (num_rows - 1 - position.y)


class GridVerseViewer:
    def __init__(self, shape: Shape, *, caption: Optional[str] = None):
        self._pos_converter = partial(convert_pos, num_rows=shape.height,)

        self._viewer_transforms = [
            rendering.Transform(translation=(1.0, 1.0)),
            rendering.Transform(scale=(0.5 / shape.width, 0.5 / shape.height),),
        ]

        m = 40
        self._viewer = rendering.Viewer(m * shape.width, m * shape.height)
        self._viewer.set_bounds(0.0, 1.0, 0.0, 1.0)

        if caption is not None:
            self._viewer.window.set_caption(caption)

        background = make_grid_background()
        self._viewer.add_geom(background)

        self._grid = make_grid(
            (0.0, 0.0), (1.0, 1.0), shape.height, shape.width,
        )

    def __del__(self):
        self._viewer.close()

    def flip_visibility(self):
        self.window.set_visible(not self.window.visible)

    @property
    def window(self) -> pyglet.window.Window:
        return self._viewer.window

    def render(self, state_or_observation: Union[State, Observation]):
        for position in state_or_observation.grid.positions():
            obj = state_or_observation.grid[position]
            if isinstance(obj, Floor):
                pass

            if isinstance(obj, Hidden):
                geom = make_hidden(obj)
                self._draw_geom_onetime(geom, position)

            if isinstance(obj, Wall):
                geom = make_wall(obj)
                self._draw_geom_onetime(geom, position)

            if isinstance(obj, Key):
                geom = make_key(obj)
                self._draw_geom_onetime(geom, position)

            if isinstance(obj, Door):
                geom = make_door(obj)
                self._draw_geom_onetime(geom, position)

            if isinstance(obj, Goal):
                geom = make_goal(obj)
                self._draw_geom_onetime(geom, position)

            if isinstance(obj, MovingObstacle):
                geom = make_moving_obstacle(obj)
                self._draw_geom_onetime(geom, position)

        geom = make_agent()
        self._draw_geom_onetime(
            geom,
            state_or_observation.agent.position,
            state_or_observation.agent.orientation,
        )

        self._viewer.add_onetime(self._grid)
        self._viewer.render(return_rgb_array=True)

    def _draw_geom_onetime(
        self,
        geom: rendering.Geom,
        position: Position,
        orientation: Orientation = Orientation.N,
    ):
        geom.add_attr(rendering.Transform(rotation=orientation.as_radians()))
        geom.add_attr(
            rendering.Transform(translation=self._pos_converter(position))
        )
        for transform in self._viewer_transforms:
            geom.add_attr(transform)
        self._viewer.add_onetime(geom)


class Controls(enum.Enum):
    PASS = 0
    QUIT = enum.auto()
    RESET = enum.auto()
    HIDE_STATE = enum.auto()
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
    print(fstr('o', Controls.CYCLE_OBSERVATION))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_path', help='env YAML file')
    parser.add_argument(
        '--fps', type=float, default=15.0, help='frames per second'
    )
    args = parser.parse_args()
    args.spf = 1 / args.fps

    with open(args.env_path) as f:
        env = make_environment(f)

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

    done = True
    while True:
        if done:
            env.reset()
            state_viewer.render(env.state)
            observation_viewer.render(env.observation)
            done = False

        command = keyboard_handler.get_command()

        if isinstance(command, Actions):
            _, done = env.step(command)

        if command is Controls.HIDE_STATE:
            state_viewer.flip_visibility()

        elif command is Controls.RESET:
            done = True
            continue

        elif command is Controls.QUIT:
            break

        elif command is Controls.CYCLE_OBSERVATION:
            name = next(observation_function_names)
            print(f'setting observation function: {name}')
            observation_function = observation_fs.factory(
                name, observation_space=env.observation_space
            )
            env.set_observation_function(observation_function)

        state_viewer.render(env.state)
        observation_viewer.render(env.observation)
        time.sleep(args.spf)


if __name__ == '__main__':
    main()
