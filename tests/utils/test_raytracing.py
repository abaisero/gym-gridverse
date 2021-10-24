import math
from typing import List

import pytest

from gym_gridverse.geometry import Area, Position
from gym_gridverse.utils.raytracing import (
    compute_ray,
    compute_rays,
    compute_rays_fancy,
)


@pytest.mark.parametrize(
    'position,area',
    [
        (Position(2, 0), Area((-1, 1), (-2, 2))),
        (Position(-2, 0), Area((-1, 1), (-2, 2))),
        (Position(0, 3), Area((-1, 1), (-2, 2))),
        (Position(0, -3), Area((-1, 1), (-2, 2))),
    ],
)
def test_compute_ray_value_error(position: Position, area: Area):
    with pytest.raises(ValueError):
        compute_ray(position, area, radians=0.0, step_size=0.01)


@pytest.mark.parametrize(
    'position,area,degrees,expected',
    [
        # from center
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            0,
            [Position(0, 0), Position(0, 1), Position(0, 2)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            45,
            [Position(0, 0), Position(1, 1)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            90,
            [Position(0, 0), Position(1, 0)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            135,
            [Position(0, 0), Position(1, -1)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            180,
            [Position(0, 0), Position(0, -1), Position(0, -2)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            225,
            [Position(0, 0), Position(-1, -1)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            270,
            [Position(0, 0), Position(-1, 0)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            315,
            [Position(0, 0), Position(-1, 1)],
        ),
        # from off-center
        (
            Position(1, 1),
            Area((-1, 1), (-2, 2)),
            0,
            [Position(1, 1), Position(1, 2)],
        ),
        (Position(1, 1), Area((-1, 1), (-2, 2)), 45, [Position(1, 1)]),
        (Position(1, 1), Area((-1, 1), (-2, 2)), 90, [Position(1, 1)]),
        (Position(1, 1), Area((-1, 1), (-2, 2)), 135, [Position(1, 1)]),
        (
            Position(1, 1),
            Area((-1, 1), (-2, 2)),
            180,
            [Position(1, 1), Position(1, 0), Position(1, -1), Position(1, -2)],
        ),
        (
            Position(1, 1),
            Area((-1, 1), (-2, 2)),
            225,
            [Position(1, 1), Position(0, 0), Position(-1, -1)],
        ),
        (
            Position(1, 1),
            Area((-1, 1), (-2, 2)),
            270,
            [Position(1, 1), Position(0, 1), Position(-1, 1)],
        ),
        (
            Position(1, 1),
            Area((-1, 1), (-2, 2)),
            315,
            [Position(1, 1), Position(0, 2)],
        ),
    ],
)
def test_compute_ray_unique(
    position: Position,
    area: Area,
    degrees: float,
    expected: List[Position],
):
    radians = degrees * math.pi / 180
    ray = compute_ray(
        position, area, radians=radians, step_size=0.01, unique=True
    )

    assert ray == expected


@pytest.mark.parametrize(
    'position,area,degrees,expected',
    [
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            0,
            [Position(0, 0), Position(0, 1), Position(0, 2)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            45,
            [Position(0, 0), Position(1, 1)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            90,
            [Position(0, 0), Position(1, 0)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            135,
            [Position(0, 0), Position(1, -1)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            180,
            [Position(0, 0), Position(0, -1), Position(0, -2)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            225,
            [Position(0, 0), Position(-1, -1)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            270,
            [Position(0, 0), Position(-1, 0)],
        ),
        (
            Position(0, 0),
            Area((-1, 1), (-2, 2)),
            315,
            [Position(0, 0), Position(-1, 1)],
        ),
    ],
)
def test_compute_ray_non_unique(
    position: Position,
    area: Area,
    degrees: float,
    expected: List[Position],
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
        (Position(-1, -2), Area((-1, 1), (-2, 2))),
        (Position(-1, 2), Area((-1, 1), (-2, 2))),
        (Position(1, -2), Area((-1, 1), (-2, 2))),
        (Position(1, 2), Area((-1, 1), (-2, 2))),
    ],
)
def test_compute_rays(position: Position, area: Area):
    rays = compute_rays(position, area)
    assert len(rays) == 360

    for ray in rays:
        assert len(ray) <= area.height + area.width - 1


@pytest.mark.parametrize(
    'position,area',
    [
        (Position(-1, -2), Area((-1, 1), (-2, 2))),
        (Position(-1, 2), Area((-1, 1), (-2, 2))),
        (Position(1, -2), Area((-1, 1), (-2, 2))),
        (Position(1, 2), Area((-1, 1), (-2, 2))),
    ],
)
def test_compute_rays_fancy(position: Position, area: Area):
    rays = compute_rays_fancy(position, area)
    assert len(rays) == (area.height + 1) * (area.width + 1)

    for ray in rays:
        assert len(ray) <= area.height + area.width - 1
