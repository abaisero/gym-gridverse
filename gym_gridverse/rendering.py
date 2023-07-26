from __future__ import annotations

import math
from functools import partial
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pyglet
from pyglet.gl import glClearColor

from gym_gridverse import rendering_gym as rendering
from gym_gridverse.action import Action
from gym_gridverse.geometry import Orientation, Position, Shape
from gym_gridverse.grid_object import (
    Beacon,
    Color,
    Door,
    Exit,
    Floor,
    GridObject,
    Hidden,
    Key,
    MovingObstacle,
    Telepod,
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


def make_grid(
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


def make_spiral(
    polar_from: Tuple[float, float], polar_to: Tuple[float, float], res: int
):
    polar = np.linspace(polar_from, polar_to, res)
    points = [(math.cos(ang) * rad, math.sin(ang) * rad) for rad, ang in polar]
    return rendering.make_polyline(points)


def make_grid_background() -> rendering.Geom:
    geom = rendering.make_polygon(
        [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
        ],
        filled=True,
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
    Color.NONE: NONE,
    Color.RED: RED,
    Color.GREEN: GREEN,
    Color.BLUE: BLUE,
    Color.YELLOW: YELLOW,
}


def make_agent() -> rendering.Geom:
    pad = 0.7
    geom_agent = rendering.make_polygon(
        [(-pad, -pad), (0.0, pad), (pad, -pad)], filled=False
    )
    geom_agent.set_linewidth(3)
    geom_agent.set_color(0, 0, 0)
    return geom_agent


def make_exit(exit_: Exit) -> rendering.Geom:
    pad = 0.8
    geom_flag = rendering.make_polyline(
        [(0.0, -pad), (0.0, pad), (pad, pad / 2), (0.0, 0.0)]
    )
    geom_flag.set_linewidth(2)
    geom_flag.add_attr(rendering.Transform(translation=(-pad / 4, 0.0)))

    if exit_.color is not Color.NONE:
        geom_exit = rendering.make_polygon(
            [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)],
            filled=True,
        )

        geom_exit.set_color(*colormap[exit_.color])
        geoms = [geom_exit, geom_flag]
    else:
        geoms = [geom_flag]

    return Group(geoms)


def make_hidden(hidden: Hidden) -> rendering.Geom:
    geom = rendering.make_polygon(
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)],
        filled=True,
    )
    return geom


def make_wall(wall: Wall) -> rendering.Geom:
    geom_background = rendering.make_polygon(
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)],
        filled=True,
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


def _make_door_open(door: Door) -> rendering.Geom:
    pad = 0.8

    geoms_frame_background = [
        rendering.make_polygon(
            [(-1.0, -1.0), (-1.0, 1.0), (-pad, 1.0), (-pad, -1.0)],
            filled=True,
        ),
        rendering.make_polygon(
            [(pad, -1.0), (pad, 1.0), (1.0, 1.0), (1.0, -1.0)],
            filled=True,
        ),
        rendering.make_polygon(
            [(-1.0, -1.0), (-1.0, -pad), (1.0, -pad), (1.0, -1.0)],
            filled=True,
        ),
        rendering.make_polygon(
            [(-1.0, pad), (-1.0, 1.0), (1.0, 1.0), (1.0, pad)],
            filled=True,
        ),
    ]
    geom_frame_background = rendering.Compound(geoms_frame_background)
    geom_frame_background.set_color(*colormap[door.color])

    geom_frame = rendering.make_polygon(
        [(-pad, -pad), (-pad, pad), (pad, pad), (pad, -pad)],
        filled=False,
    )

    return Group([geom_frame_background, geom_frame])


def _make_door_closed_locked(door: Door) -> rendering.Geom:
    geom_door = rendering.make_polygon(
        # [(-0.8, -0.8), (-0.8, 0.8), (0.8, 0.8), (0.8, -0.8)], filled=True,
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)],
        filled=True,
    )
    geom_door.set_color(*colormap[door.color])

    pad = 0.8
    geom_frame = rendering.make_polygon(
        [(-pad, -pad), (-pad, pad), (pad, pad), (pad, -pad)],
        filled=False,
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


def _make_door_closed_unlocked(door: Door) -> rendering.Geom:
    geom_door = rendering.make_polygon(
        # [(-0.8, -0.8), (-0.8, 0.8), (0.8, 0.8), (0.8, -0.8)], filled=True,
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)],
        filled=True,
    )
    geom_door.set_color(*colormap[door.color])

    pad = 0.8
    geom_frame = rendering.make_polygon(
        [(-pad, -pad), (-pad, pad), (pad, pad), (pad, -pad)],
        filled=False,
    )

    geom_handle = rendering.make_circle(radius=0.2, res=10, filled=False)
    geom_handle.add_attr(rendering.Transform(translation=(0.4, 0.0)))

    return Group([geom_door, geom_frame, geom_handle])


def make_door(door: Door) -> rendering.Geom:
    return (
        _make_door_open(door)
        if door.is_open
        else _make_door_closed_locked(door)
        if door.is_locked
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


def make_key(key: Key) -> rendering.Geom:
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


def make_moving_obstacle(obstacle: MovingObstacle) -> rendering.Geom:
    pad = 0.8
    geom = rendering.make_polygon(
        [
            (-pad, 0.0),
            (0.0, pad),
            (pad, 0.0),
            (0.0, -pad),
        ],
        filled=True,
    )
    geom.set_color(*RED)

    geom_outline = rendering.make_polygon(
        [
            (-pad, 0.0),
            (0.0, pad),
            (pad, 0.0),
            (0.0, -pad),
        ],
        filled=False,
    )
    geom_outline.set_linewidth(3)

    return Group([geom, geom_outline])


def make_telepod(telepod: Telepod) -> rendering.Geom:
    res = 100
    geom_circle = rendering.make_circle(0.8, res=res, filled=True)
    geom_circle.set_color(*colormap[telepod.color])
    geom_boundary = rendering.make_circle(0.8, res=res, filled=False)
    geom_boundary.set_linewidth(2)
    geom_spiral = make_spiral((0.8, 0.0), (0.0, 4 * math.pi), res)
    geom_spiral.set_linewidth(2)
    return Group([geom_circle, geom_boundary, geom_spiral])


def make_beacon(beacon: Beacon) -> rendering.Geom:
    res = 100
    geom_circle = rendering.make_circle(0.8, res=res, filled=True)
    geom_circle.set_color(*colormap[beacon.color])
    geom_boundary = rendering.make_circle(0.8, res=res, filled=False)
    geom_boundary.set_linewidth(2)
    geom_diag_1 = rendering.make_polygon(
        [(0.4, -0.4), (-0.4, 0.4)], filled=False
    )
    geom_diag_1.set_linewidth(2)
    geom_diag_2 = rendering.make_polygon(
        [(0.4, 0.4), (-0.4, -0.4)], filled=False
    )
    geom_diag_2.set_linewidth(2)

    return Group([geom_circle, geom_boundary, geom_diag_1, geom_diag_2])


def make_unknown(obj: GridObject) -> rendering.Geom:
    res = 100
    geom_circle = rendering.make_circle(0.8, res=res, filled=True)
    geom_circle.set_color(*colormap[obj.color])
    geom_boundary = rendering.make_circle(0.8, res=res, filled=False)
    geom_boundary.set_linewidth(2)
    geom_diag_1 = rendering.make_polygon(
        [(0.4, -0.4), (-0.4, 0.4)], filled=False
    )
    geom_diag_1.set_linewidth(2)
    geom_diag_2 = rendering.make_polygon(
        [(0.4, 0.4), (-0.4, -0.4)], filled=False
    )
    geom_diag_2.set_linewidth(2)

    return Group([geom_circle, geom_boundary, geom_diag_1, geom_diag_2])


def convert_pos(position: Position, *, num_rows: int) -> Tuple[float, float]:
    return 2 * position.x, 2 * (num_rows - 1 - position.y)


# TODO: clean this code;  this is barely working
class _CustomViewer(rendering.Viewer):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.transform = rendering.Transform()

        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(
            pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA
        )

    def render(self, return_rgb_array=False, *, other_drawables=[]):
        glClearColor(1, 1, 1, 1)
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.clear()

        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()

        for drawable in other_drawables:
            drawable.draw()

        arr = None
        if return_rgb_array:
            buff = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buff.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buff.height, buff.width, 4)
            arr = arr[::-1, :, 0:3]

        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen


class GridVerseViewer:
    def __init__(self, shape: Shape, *, caption: Optional[str] = None):
        self._pos_converter = partial(
            convert_pos,
            num_rows=shape.height,
        )

        self._viewer_transforms = [
            rendering.Transform(translation=(1.0, 1.0)),
            rendering.Transform(scale=(0.5 / shape.width, 0.5 / shape.height)),
        ]

        m = 40
        self._viewer = _CustomViewer(m * shape.width, m * shape.height)
        self._viewer.set_bounds(0.0, 1.0, 0.0, 1.0)

        if caption is not None:
            self._viewer.window.set_caption(caption)

        background = make_grid_background()
        self._viewer.add_geom(background)

        self._grid = make_grid(
            (0.0, 0.0),
            (1.0, 1.0),
            shape.height,
            shape.width,
        )

        self._draw_hud = True
        self._hud_format = (
            'action: {action}'
            '\nreward: {reward}'
            '\nreturn: {ret}'
            '\ndone: {done}'
        )
        self._hud_document = pyglet.text.document.UnformattedDocument()
        self._hud_document.set_style(
            None,
            None,
            {'color': (255, 255, 255, 255), 'background_color': (0, 0, 0, 100)},
        )
        self._hud_layout = pyglet.text.layout.TextLayout(
            self._hud_document,
            self.window.width,
            self.window.height,
            multiline=True,
        )
        self._hud_layout.x = 0
        self._hud_layout.y = (
            self._viewer.height
        )  # window.height uses the first window?
        self._hud_layout.anchor_x = 'left'
        self._hud_layout.anchor_y = 'top'

    def _update_hud(
        self,
        *,
        action: Optional[Action] = None,
        reward: Optional[float] = None,
        ret: Optional[float] = None,
        done: Optional[bool] = None,
    ):
        self._hud_document.text = self._hud_format.format(
            action='' if action is None else action.name,
            reward='' if reward is None else f'{reward:-.2f}',
            ret='' if ret is None else f'{ret:-.2f}',
            done='' if done is None else done,
        )

    def __del__(self):
        self.close()

    def close(self):
        self._viewer.close()

    def flip_visibility(self):
        self.window.set_visible(not self.window.visible)

    def flip_hud(self):
        self._draw_hud = not self._draw_hud

    @property
    def window(self) -> pyglet.window.Window:
        return self._viewer.window

    def render(
        self,
        state_or_observation: Union[State, Observation],
        *,
        action: Optional[Action] = None,
        reward: Optional[float] = None,
        ret: Optional[float] = None,
        done: Optional[bool] = None,
        return_rgb_array: bool = False,
    ):
        self._update_hud(action=action, reward=reward, ret=ret, done=done)

        for position in state_or_observation.grid.area.positions():
            obj = state_or_observation.grid[position]

            if isinstance(obj, Floor):
                pass

            elif isinstance(obj, Hidden):
                geom = make_hidden(obj)
                self._draw_geom_onetime(geom, position)

            elif isinstance(obj, Wall):
                geom = make_wall(obj)
                self._draw_geom_onetime(geom, position)

            elif isinstance(obj, Key):
                geom = make_key(obj)
                self._draw_geom_onetime(geom, position)

            elif isinstance(obj, Door):
                geom = make_door(obj)
                self._draw_geom_onetime(geom, position)

            elif isinstance(obj, Exit):
                geom = make_exit(obj)
                self._draw_geom_onetime(geom, position)

            elif isinstance(obj, MovingObstacle):
                geom = make_moving_obstacle(obj)
                self._draw_geom_onetime(geom, position)

            elif isinstance(obj, Telepod):
                geom = make_telepod(obj)
                self._draw_geom_onetime(geom, position)

            elif isinstance(obj, Beacon):
                geom = make_beacon(obj)
                self._draw_geom_onetime(geom, position)

            else:
                # unknown grid object
                geom = make_unknown(obj)
                self._draw_geom_onetime(geom, position)

        geom = make_agent()
        self._draw_geom_onetime(
            geom,
            state_or_observation.agent.position,
            state_or_observation.agent.orientation,
        )

        self._viewer.add_onetime(self._grid)
        other_drawables = [self._hud_layout] if self._draw_hud else []
        return self._viewer.render(
            return_rgb_array=return_rgb_array, other_drawables=other_drawables
        )

    def _draw_geom_onetime(
        self,
        geom: rendering.Geom,
        position: Position,
        orientation: Orientation = Orientation.F,
    ):
        rotation = _orientation_as_radians[orientation]
        geom.add_attr(rendering.Transform(rotation=rotation))
        geom.add_attr(
            rendering.Transform(translation=self._pos_converter(position))
        )
        for transform in self._viewer_transforms:
            geom.add_attr(transform)
        self._viewer.add_onetime(geom)


_orientation_as_radians = {
    Orientation.F: 0.0,
    Orientation.L: math.pi / 2,
    Orientation.B: math.pi,
    Orientation.R: math.pi * 3 / 2,
}
