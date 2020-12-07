import math
from typing import List

import pytest

from gym_gridverse.geometry import Area, PositionOrTuple
from gym_gridverse.utils.raytracing import (
    compute_ray,
    compute_rays,
    compute_rays_fancy,
)


@pytest.mark.parametrize(
    'position,area',
    [
        ((2, 0), Area((-1, 1), (-2, 2))),
        ((-2, 0), Area((-1, 1), (-2, 2))),
        ((0, 3), Area((-1, 1), (-2, 2))),
        ((0, -3), Area((-1, 1), (-2, 2))),
    ],
)
def test_compute_ray_value_error(position: PositionOrTuple, area: Area):
    with pytest.raises(ValueError):
        compute_ray(position, area, radians=0.0, step_size=0.01)


@pytest.mark.parametrize(
    'position,area,degrees,expected',
    [
        # from center
        ((0, 0), Area((-1, 1), (-2, 2)), 0, [(0, 0), (0, 1), (0, 2)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 45, [(0, 0), (1, 1)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 90, [(0, 0), (1, 0)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 135, [(0, 0), (1, -1)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 180, [(0, 0), (0, -1), (0, -2)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 225, [(0, 0), (-1, -1)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 270, [(0, 0), (-1, 0)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 315, [(0, 0), (-1, 1)]),
        # from off-center
        ((1, 1), Area((-1, 1), (-2, 2)), 0, [(1, 1), (1, 2)]),
        ((1, 1), Area((-1, 1), (-2, 2)), 45, [(1, 1)]),
        ((1, 1), Area((-1, 1), (-2, 2)), 90, [(1, 1)]),
        ((1, 1), Area((-1, 1), (-2, 2)), 135, [(1, 1)]),
        (
            (1, 1),
            Area((-1, 1), (-2, 2)),
            180,
            [(1, 1), (1, 0), (1, -1), (1, -2)],
        ),
        (
            (1, 1),
            Area((-1, 1), (-2, 2)),
            225,
            [(1, 1), (0, 0), (-1, -1)],
        ),
        (
            (1, 1),
            Area((-1, 1), (-2, 2)),
            270,
            [(1, 1), (0, 1), (-1, 1)],
        ),
        (
            (1, 1),
            Area((-1, 1), (-2, 2)),
            315,
            [(1, 1), (0, 2)],
        ),
    ],
)
def test_compute_ray_unique(
    position: PositionOrTuple,
    area: Area,
    degrees: float,
    expected: List[PositionOrTuple],
):
    radians = degrees * math.pi / 180
    ray = compute_ray(
        position, area, radians=radians, step_size=0.01, unique=True
    )

    assert ray == expected


@pytest.mark.parametrize(
    'position,area,degrees,expected',
    [
        ((0, 0), Area((-1, 1), (-2, 2)), 0, [(0, 0), (0, 1), (0, 2)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 45, [(0, 0), (1, 1)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 90, [(0, 0), (1, 0)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 135, [(0, 0), (1, -1)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 180, [(0, 0), (0, -1), (0, -2)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 225, [(0, 0), (-1, -1)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 270, [(0, 0), (-1, 0)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 315, [(0, 0), (-1, 1)]),
    ],
)
def test_compute_ray_non_unique(
    position: PositionOrTuple,
    area: Area,
    degrees: float,
    expected: List[PositionOrTuple],
):
    radians = degrees * math.pi / 180
    ray = compute_ray(
        position, area, radians=radians, step_size=0.01, unique=False
    )

    assert len(ray) > len(expected)
    assert set(expected).issubset(ray)


@pytest.mark.parametrize(
    'position,area',
    [
        ((-1, -2), Area((-1, 1), (-2, 2))),
        ((-1, 2), Area((-1, 1), (-2, 2))),
        ((1, -2), Area((-1, 1), (-2, 2))),
        ((1, 2), Area((-1, 1), (-2, 2))),
    ],
)
def test_compute_rays(position: PositionOrTuple, area: Area):
    rays = compute_rays(position, area)
    assert len(rays) == 360

    for ray in rays:
        assert len(ray) <= area.height + area.width - 1


@pytest.mark.parametrize(
    'position,area',
    [
        ((-1, -2), Area((-1, 1), (-2, 2))),
        ((-1, 2), Area((-1, 1), (-2, 2))),
        ((1, -2), Area((-1, 1), (-2, 2))),
        ((1, 2), Area((-1, 1), (-2, 2))),
    ],
)
def test_compute_rays_fancy(position: PositionOrTuple, area: Area):
    rays = compute_rays_fancy(position, area)
    assert len(rays) == (area.height + 1) * (area.width + 1)

    for ray in rays:
        assert len(ray) <= area.height + area.width - 1
