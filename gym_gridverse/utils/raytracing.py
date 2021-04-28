import itertools as itt
import math
from functools import lru_cache
from typing import Iterable, List

import more_itertools as mitt
import numpy as np

from gym_gridverse.debugging import checkraise
from gym_gridverse.geometry import Area, Position, PositionOrTuple

Ray = List[Position]


def compute_ray(
    position: PositionOrTuple,
    area: Area,
    *,
    radians: float,
    step_size: float,
    unique: bool = True,
) -> Ray:
    """Returns a ray from a given position.

    A ray is a list of positions which are hit by a direct line starting at the
    center of the given position and moving along the given direction (in
    radians) until the area is left.

    Args:
        position (PositionOrTuple): initial position, must be in area.
        area (Area): boundary over rays.
        radians (float): ray direction.
        step_size (float): ray step granularity.
        unique (bool): If true, the same position can appear twice in the ray.

    Returns:
        Ray: ray from the given position until the area boundary
    """

    checkraise(
        lambda: area.contains(position),
        ValueError,
        'position {} must be inside area {}',
        position,
        area,
    )

    position = Position.from_position_or_tuple(position)

    y0, x0 = float(position.y), float(position.x)
    dy = step_size * math.sin(radians)
    dx = step_size * math.cos(radians)

    ys = (y0 + i * dy for i in itt.count())
    xs = (x0 + i * dx for i in itt.count())
    positions: Iterable[Position]
    positions = (Position(round(y), round(x)) for y, x in zip(ys, xs))
    positions = itt.takewhile(area.contains, positions)
    positions = mitt.unique_everseen(positions) if unique else positions

    return list(positions)


def compute_rays(position: PositionOrTuple, area: Area) -> List[Ray]:
    """Returns rays obtained at 1° granularity.

    A ray is a list of positions which are hit by a direct line starting at the
    center of the given position and moving along a direction until the area is
    left.  This method will search for the ingeter directions between 0° and
    359°.

    Args:
        position (PositionOrTuple): initial position, must be in area.
        area (Area): boundary over rays.

    Returns:
        List[Ray]:
    """
    rays: List[Ray] = []

    radians_over_degrees = math.pi / 180.0
    degrees = range(360)
    radians = (deg * radians_over_degrees for deg in degrees)
    rays = [
        compute_ray(position, area, radians=rad, step_size=0.01)
        for rad in radians
    ]

    return rays


def compute_rays_fancy(position: PositionOrTuple, area: Area) -> List[Ray]:
    """Returns rays obtained by targeting edge points.

    A ray is a list of positions which are hit by a direct line starting at the
    center of the given position and moving along a direction until the area is
    left.  This method will search in the directions towards all other cell
    edges.

    Args:
        position (PositionOrTuple): initial position, must be in area.
        area (Area): boundary over rays.

    Returns:
        List[Ray]:
    """
    position = Position.from_position_or_tuple(position)

    # compute corners of each cell
    ys = np.linspace(area.ymin, area.ymax + 1, num=area.height + 1) - 0.5
    xs = np.linspace(area.xmin, area.xmax + 1, num=area.width + 1) - 0.5

    # center all positions
    ys = ys - position.y
    xs = xs - position.x

    # compute points and angles
    yys, xxs = np.meshgrid(ys, xs)
    radians = np.arctan2(yys, xxs)
    radians = np.sort(radians, axis=None)

    rays = [
        compute_ray(position, area, radians=rad, step_size=0.01)
        for rad in radians
    ]

    return rays


# the ray functions are deterministic and can be cached for efficiency (extra
# calls for python3.7 compatibility)
cached_compute_rays = lru_cache()(compute_rays)
cached_compute_rays_fancy = lru_cache()(compute_rays_fancy)
