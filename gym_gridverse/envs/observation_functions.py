import math
import random
from functools import lru_cache
from typing import Callable, Iterator, List

import more_itertools as mitt
import numpy as np

from gym_gridverse.geometry import Area, Orientation, Position
from gym_gridverse.grid_object import Hidden
from gym_gridverse.info import Agent
from gym_gridverse.observation import Observation
from gym_gridverse.state import State

ObservationFunction = Callable[[State], Observation]

# TODO write documentation


def minigrid_observation(state: State) -> Observation:
    area = state.agent.get_pov_area()
    grid = state.grid.subgrid(area).change_orientation(state.agent.orientation)

    visibility_mask = np.zeros((area.height, area.width), dtype=bool)
    visibility_mask[area.height - 1, area.width // 2] = True  # agent

    for y in range(area.height - 1, -1, -1):
        for x in range(area.width - 1):
            if visibility_mask[y, x] and grid[Position(y, x)].transparent:
                visibility_mask[y, x + 1] = True
                if y > 0:
                    visibility_mask[y - 1, x] = True
                    visibility_mask[y - 1, x + 1] = True

        for x in range(area.width - 1, 0, -1):
            if visibility_mask[y, x] and grid[Position(y, x)].transparent:
                visibility_mask[y, x - 1] = True
                if y > 0:
                    visibility_mask[y - 1, x] = True
                    visibility_mask[y - 1, x - 1] = True

    for y in range(area.height):
        for x in range(area.width):
            if not visibility_mask[y, x]:
                grid[Position(y, x)] = Hidden()

    agent = Agent(Position(-1, -1), Orientation.N, state.agent.obj)
    return Observation(grid, agent)


# TODO test this
# TODO cache this?
def ray_positions(
    start_pos: Position, area: Area, radians: float, step_size: float
) -> Iterator[Position]:
    y, x = float(start_pos.y), float(start_pos.x)
    dy = -math.sin(radians)
    dx = math.cos(radians)

    pos = start_pos
    while area.contains(pos):
        yield pos

        y += step_size * dy
        x += step_size * dx
        pos = Position(round(y), round(x))


# TODO test this
@lru_cache()
def rays_positions(start_pos: Position, area: Area) -> List[List[Position]]:
    rays: List[List[Position]] = []

    step_size = 0.01
    for degrees in range(360):
        # conversion to radians
        radians = degrees * math.pi / 180.0
        ray = ray_positions(start_pos, area, radians, step_size)
        ray = mitt.unique_justseen(ray)
        rays.append(list(ray))

    return rays


def raytracing_observation(state: State) -> Observation:
    area = state.agent.get_pov_area()
    grid = state.grid.subgrid(area).change_orientation(state.agent.orientation)

    pos = Position(6, 3)  # TODO use observation shape
    area = Area((0, 6), (0, 6))  # TODO use observation shape
    rays = rays_positions(pos, area)

    counts_n = np.zeros((area.height, area.width), dtype=int)
    counts_d = np.zeros((area.height, area.width), dtype=int)

    for ray in rays:
        light = True
        for pos in ray:
            if light:
                counts_n[pos.y, pos.x] += 1

            counts_d[pos.y, pos.x] += 1

            light = light and grid[pos].transparent

    mask = counts_n > 0  # at least one ray makes it
    # mask = counts_n > 0.5 * counts_d # half of the rays make it
    # mask = counts_n > 0.1 * counts_d  # 10% of the rays make it
    # mask = counts_n > 1  # at least 2 rays make it

    for y in range(area.height):
        for x in range(area.width):
            if not mask[y, x]:
                grid[Position(y, x)] = Hidden()

    agent = Agent(Position(-1, -1), Orientation.N, state.agent.obj)
    return Observation(grid, agent)


def stochastic_raytracing_observation(state: State) -> Observation:
    area = state.agent.get_pov_area()
    grid = state.grid.subgrid(area).change_orientation(state.agent.orientation)

    pos = Position(6, 3)
    area = Area((0, 6), (0, 6))
    rays = rays_positions(pos, area)

    counts_n = np.zeros((area.height, area.width), dtype=int)
    counts_d = np.zeros((area.height, area.width), dtype=int)

    for ray in rays:
        light = True
        for pos in ray:
            if light:
                counts_n[pos.y, pos.x] += 1

            counts_d[pos.y, pos.x] += 1

            light = light and grid[pos].transparent

    probs = np.nan_to_num(counts_n / counts_d)

    # TODO it's technically possible to observe a cell behind a hidden cell now
    # to fix this we'd have to take the rays into account again..
    # might be too much hassle though;  we wouldn't be able to separate the
    # ray-tracing from the masking

    for y in range(area.height):
        for x in range(area.width):
            if random.random() > probs[y, x]:
                grid[Position(y, x)] = Hidden()

    agent = Agent(Position(-1, -1), Orientation.N, state.agent.obj)
    return Observation(grid, agent)
