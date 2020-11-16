import math
from functools import lru_cache, partial
from typing import Callable, Iterator, List, Optional

import more_itertools as mitt
import numpy as np
import numpy.random as rnd

from gym_gridverse.geometry import Area, Orientation, Position
from gym_gridverse.grid_object import Hidden
from gym_gridverse.info import Agent, Grid
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import ObservationSpace
from gym_gridverse.state import State

ObservationFunction = Callable[[State], Observation]
VisibilityFunction = Callable[[Grid, Position], np.array]

# TODO write documentation


def full_visibility(state: State, *, observation_space: ObservationSpace):
    area = state.agent.get_pov_area(observation_space.area)
    observation_grid = state.grid.subgrid(area).change_orientation(
        state.agent.orientation
    )

    observation_agent = Agent(
        observation_space.agent_position, Orientation.N, state.agent.obj
    )
    return Observation(observation_grid, observation_agent)


def from_visibility(
    state: State,
    *,
    observation_space: ObservationSpace,
    visibility_function: VisibilityFunction,
):
    area = state.agent.get_pov_area(observation_space.area)
    observation_grid = state.grid.subgrid(area).change_orientation(
        state.agent.orientation
    )
    visibility = visibility_function(
        observation_grid, observation_space.agent_position
    )

    if visibility.shape != observation_space.grid_shape:
        raise ValueError('incorrect shape of visibility mask')

    for pos in observation_grid.positions():
        if not visibility[pos.y, pos.x]:
            observation_grid[pos] = Hidden()

    observation_agent = Agent(
        observation_space.agent_position, Orientation.N, state.agent.obj
    )
    return Observation(observation_grid, observation_agent)


def minigrid_visibility(grid: Grid, position: Position) -> np.ndarray:
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


# TODO test this
def ray_positions(
    start_pos: Position, area: Area, *, radians: float, step_size: float
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

    for degrees in range(360):
        # conversion to radians
        radians = degrees * math.pi / 180.0
        ray = ray_positions(start_pos, area, radians=radians, step_size=0.01)
        ray = mitt.unique_justseen(ray)
        rays.append(list(ray))

    return rays


def raytracing_visibility(grid: Grid, position: Position) -> np.ndarray:
    area = Area((0, grid.height - 1), (0, grid.width - 1))
    rays = rays_positions(position, area)

    counts_num = np.zeros((area.height, area.width), dtype=int)
    counts_den = np.zeros((area.height, area.width), dtype=int)

    for ray in rays:
        light = True
        for pos in ray:
            if light:
                counts_num[pos.y, pos.x] += 1

            counts_den[pos.y, pos.x] += 1

            light = light and grid[pos].transparent

    visibility = counts_num > 0  # at least one ray makes it
    # visibility = counts_num > 0.5 * counts_den # half of the rays make it
    # visibility = counts_num > 0.1 * counts_den  # 10% of the rays make it
    # visibility = counts_num > 1  # at least 2 rays make it

    return visibility


def stochastic_raytracing_visibility(
    grid: Grid, position: Position
) -> np.ndarray:
    area = Area((0, grid.height - 1), (0, grid.width - 1))
    rays = rays_positions(position, area)

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
    visibility = probs <= rnd.random((probs.shape))  # pylint: disable=no-member
    return visibility


minigrid_observation = partial(
    from_visibility, visibility_function=minigrid_visibility
)

raytracing_observation = partial(
    from_visibility, visibility_function=raytracing_visibility
)

stochastic_raytracing_observation = partial(
    from_visibility, visibility_function=raytracing_visibility
)


def factory(
    name: str,
    *,
    observation_space: Optional[ObservationSpace] = None,
    visibility_function: Optional[VisibilityFunction] = None,
) -> ObservationFunction:

    if name == 'full_visibility':
        if None in [observation_space]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(full_visibility, observation_space=observation_space)

    if name == 'from_visibility':
        if None in [observation_space, visibility_function]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            from_visibility,
            observation_space=observation_space,
            visibility_function=visibility_function,
        )

    if name == 'minigrid_observation':
        if None in [observation_space]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            minigrid_observation, observation_space=observation_space
        )

    if name == 'raytracing_observation':
        if None in [observation_space]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            raytracing_observation, observation_space=observation_space
        )

    if name == 'stochastic_raytracing_observation':
        if None in [observation_space]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            stochastic_raytracing_observation,
            observation_space=observation_space,
        )

    raise ValueError(f'invalid observation function name {name}')
