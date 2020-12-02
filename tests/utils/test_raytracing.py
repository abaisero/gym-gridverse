import math
from typing import List

import pytest

from gym_gridverse.geometry import Area, PositionOrTuple
from gym_gridverse.utils.raytracing import compute_ray


@pytest.mark.parametrize(
    'start_pos,area,degrees,expected',
    [
        ((0, 0), Area((-1, 1), (-2, 2)), 0, [(0, 0), (0, 1), (0, 2)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 90, [(0, 0), (-1, 0)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 180, [(0, 0), (0, -1), (0, -2)]),
        ((0, 0), Area((-1, 1), (-2, 2)), 270, [(0, 0), (1, 0)]),
    ],
)
def test_compute_ray_unique(
    start_pos: PositionOrTuple,
    area: Area,
    degrees: float,
    expected: List[PositionOrTuple],
):
    radians = degrees * math.pi / 180
    ray = compute_ray(
        start_pos, area, radians=radians, step_size=0.01, unique=True
    )

    assert ray == expected
