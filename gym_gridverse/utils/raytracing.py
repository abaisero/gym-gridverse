import itertools as itt
import math
from functools import lru_cache
from typing import Iterable, List

import more_itertools as mitt
import numpy as np

from gym_gridverse.geometry import Area, Position, PositionOrTuple

Ray = List[Position]


# TODO test this
def compute_ray(
    start_pos: PositionOrTuple,
    area: Area,
    *,
    radians: float,
    step_size: float,
    unique: bool
) -> List[Position]:
    """returns ray of positions"""

    start_pos = Position.from_position_or_tuple(start_pos)

    y0, x0 = float(start_pos.y), float(start_pos.x)
    dy = step_size * -math.sin(radians)
    dx = step_size * math.cos(radians)

    ys = (y0 + i * dy for i in itt.count())
    xs = (x0 + i * dx for i in itt.count())
    positions: Iterable[Position]
    positions = (Position(round(y), round(x)) for y, x in zip(ys, xs))
    positions = itt.takewhile(area.contains, positions)
    positions = mitt.unique_everseen(positions) if unique else positions

    return list(positions)


# TODO test this
@lru_cache()
def compute_rays(start_pos: Position, area: Area) -> List[List[Position]]:
    rays: List[Ray] = []

    radians_over_degrees = math.pi / 180.0
    degrees = range(360)
    radians = (deg * radians_over_degrees for deg in degrees)
    rays = [
        compute_ray(start_pos, area, radians=rad, step_size=0.01, unique=True)
        for rad in radians
    ]

    return rays


# TODO test this
@lru_cache()
def compute_rays_fancy(start_pos: Position, area: Area) -> List[List[Position]]:
    rays: List[Ray] = []

    # compute corners of each cell
    ys = np.linspace(area.ymin, area.ymax + 1, num=area.height + 1) - 0.5
    xs = np.linspace(area.xmin, area.xmax + 1, num=area.width + 1) - 0.5

    #  re-center and flip y axis
    ys = start_pos.y - ys
    xs = xs - start_pos.x

    # compute points and angles
    yys, xxs = np.meshgrid(ys, xs)
    radians = np.arctan2(yys, xxs)
    radians = np.sort(radians, axis=None)

    rays = [
        compute_ray(start_pos, area, radians=rad, step_size=0.01, unique=True)
        for rad in radians
    ]

    return rays
