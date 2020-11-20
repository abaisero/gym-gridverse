from functools import partial
from typing import Optional

import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.envs.visibility_functions import (
    VisibilityFunction,
    full_visibility,
    minigrid_visibility,
    raytracing_visibility,
    stochastic_raytracing_visibility,
)
from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import Hidden
from gym_gridverse.info import Agent
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import ObservationSpace
from gym_gridverse.state import State


class ObservationFunction(Protocol):
    def __call__(
        self, state: State, *, rng: Optional[rnd.Generator] = None
    ) -> Observation:
        ...


# TODO write documentation


def from_visibility(
    state: State,
    *,
    observation_space: ObservationSpace,
    visibility_function: VisibilityFunction,
    rng: Optional[rnd.Generator] = None,
):
    area = state.agent.get_pov_area(observation_space.area)
    observation_grid = state.grid.subgrid(area).change_orientation(
        state.agent.orientation
    )
    visibility = visibility_function(
        observation_grid, observation_space.agent_position, rng=rng
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


full_observation = partial(from_visibility, visibility_function=full_visibility)

minigrid_observation = partial(
    from_visibility, visibility_function=minigrid_visibility
)

raytracing_observation = partial(
    from_visibility, visibility_function=raytracing_visibility
)

stochastic_raytracing_observation = partial(
    from_visibility,
    visibility_function=stochastic_raytracing_visibility,
)


def factory(
    name: str,
    *,
    observation_space: Optional[ObservationSpace] = None,
    visibility_function: Optional[VisibilityFunction] = None,
) -> ObservationFunction:

    if name == 'from_visibility':
        if None in [observation_space, visibility_function]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            from_visibility,
            observation_space=observation_space,
            visibility_function=visibility_function,
        )

    if name == 'full_observation':
        if None in [observation_space]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(full_observation, observation_space=observation_space)

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
