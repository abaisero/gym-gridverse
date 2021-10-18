from typing import Optional

import numpy as np
import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.geometry import (
    Area,
    Position,
    StrideDirection,
    diagonal_strides,
)
from gym_gridverse.grid import Grid
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.utils.raytracing import cached_compute_rays_fancy


class VisibilityFunction(Protocol):
    def __call__(
        self,
        grid: Grid,
        position: Position,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> np.ndarray:
        ...


def full_visibility(
    grid: Grid,
    position: Position,  # pylint: disable = unused-argument
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable = unused-argument
) -> np.ndarray:

    return np.ones((grid.height, grid.width), dtype=bool)


def partial_visibility(
    grid: Grid,
    position: Position,
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
) -> np.ndarray:

    if position.y != grid.height - 1:
        #  gym-minigrid does not handle this case, and we are not currently
        #  generalizing it
        raise NotImplementedError

    visibility = np.zeros((grid.height, grid.width), dtype=bool)
    visibility[position.y, position.x] = True  # agent

    # front
    x = position.x
    for y in range(position.y - 1, -1, -1):
        visibility[y, x] = visibility[y + 1, x] and grid[y + 1, x].transparent

    # right
    y = position.y
    for x in range(position.x + 1, grid.width):
        visibility[y, x] = visibility[y, x - 1] and grid[y, x - 1].transparent

    # left
    y = position.y
    for x in range(position.x - 1, -1, -1):
        visibility[y, x] = visibility[y, x + 1] and grid[y, x + 1].transparent

    # top left
    positions = diagonal_strides(
        Area(
            (0, position.y - 1),
            (0, position.x - 1),
        ),
        StrideDirection.NW,
    )
    for p in positions:
        visibility[p.y, p.x] = (
            (grid[p.y + 1, p.x].transparent and visibility[p.y + 1, p.x])
            or (grid[p.y, p.x + 1].transparent and visibility[p.y, p.x + 1])
            or (
                grid[p.y + 1, p.x + 1].transparent
                and visibility[p.y + 1, p.x + 1]
            )
        )

    # top right
    positions = diagonal_strides(
        Area(
            (0, position.y - 1),
            (position.x + 1, grid.width - 1),
        ),
        StrideDirection.NE,
    )
    for p in positions:
        visibility[p.y, p.x] = (
            (grid[p.y + 1, p.x].transparent and visibility[p.y + 1, p.x])
            or (grid[p.y, p.x - 1].transparent and visibility[p.y, p.x - 1])
            or (
                grid[p.y + 1, p.x - 1].transparent
                and visibility[p.y + 1, p.x - 1]
            )
        )

    return visibility


def minigrid_visibility(
    grid: Grid,
    position: Position,
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable = unused-argument
) -> np.ndarray:

    if position.y != grid.height - 1:
        #  gym-minigrid does not handle this case, and we are not currently
        #  generalizing it
        raise NotImplementedError

    visibility = np.zeros((grid.height, grid.width), dtype=bool)
    visibility[position.y, position.x] = True  # agent

    for y in range(grid.height - 1, -1, -1):
        for x in range(grid.width - 1):
            if visibility[y, x] and grid[y, x].transparent:
                visibility[y, x + 1] = True
                if y > 0:
                    visibility[y - 1, x] = True
                    visibility[y - 1, x + 1] = True

        for x in range(grid.width - 1, 0, -1):
            if visibility[y, x] and grid[y, x].transparent:
                visibility[y, x - 1] = True
                if y > 0:
                    visibility[y - 1, x] = True
                    visibility[y - 1, x - 1] = True

    return visibility


def raytracing_visibility(
    grid: Grid,
    position: Position,
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
) -> np.ndarray:

    area = Area((0, grid.height - 1), (0, grid.width - 1))
    rays = cached_compute_rays_fancy(position, area)

    counts_num = np.zeros((area.height, area.width), dtype=int)
    counts_den = np.zeros((area.height, area.width), dtype=int)

    for ray in rays:
        light = True
        for pos in ray:
            if light:
                counts_num[pos.y, pos.x] += 1

            counts_den[pos.y, pos.x] += 1

            light = light and grid[pos].transparent

    # TODO: add as parameter to function
    visibility = counts_num > 0  # at least one ray makes it
    # visibility = counts_num > 0.5 * counts_den # half of the rays make it
    # visibility = counts_num > 0.1 * counts_den  # 10% of the rays make it
    # visibility = counts_num > 1  # at least 2 rays make it

    return visibility


def stochastic_raytracing_visibility(
    grid: Grid,
    position: Position,
    *,
    rng: Optional[rnd.Generator] = None,
) -> np.ndarray:
    # TODO: add test
    rng = get_gv_rng_if_none(rng)

    area = Area((0, grid.height - 1), (0, grid.width - 1))
    rays = cached_compute_rays_fancy(position, area)

    counts_num = np.zeros((area.height, area.width), dtype=int)
    counts_den = np.zeros((area.height, area.width), dtype=int)

    for ray in rays:
        light = True
        for pos in ray:
            if light:
                counts_num[pos.y, pos.x] += 1

            counts_den[pos.y, pos.x] += 1

            light = light and grid[pos].transparent

    probs = np.nan_to_num(counts_num / counts_den)
    visibility = probs <= rng.random(probs.shape)
    return visibility


def factory(name: str) -> VisibilityFunction:

    if name == 'full_visibility':
        return full_visibility

    if name == 'partial_visibility':
        return partial_visibility

    if name == 'minigrid_visibility':
        return minigrid_visibility

    if name == 'raytracing_visibility':
        return raytracing_visibility

    if name == 'stochastic_raytracing_visibility':
        return stochastic_raytracing_visibility

    raise ValueError(f'invalid visibility function name {name}')
