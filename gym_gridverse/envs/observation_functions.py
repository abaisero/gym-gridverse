from functools import partial
from typing import Optional

import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.agent import Agent
from gym_gridverse.debugging import checkraise
from gym_gridverse.envs.visibility_functions import (
    VisibilityFunction,
    full_visibility,
    minigrid_visibility,
    partial_visibility,
    raytracing_visibility,
    stochastic_raytracing_visibility,
)
from gym_gridverse.geometry import Orientation, Shape
from gym_gridverse.grid_object import Hidden
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import ObservationSpace
from gym_gridverse.state import State


class ObservationFunction(Protocol):
    def __call__(
        self, state: State, *, rng: Optional[rnd.Generator] = None
    ) -> Observation:
        ...


# TODO: write documentation


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

    checkraise(
        lambda: Shape(*visibility.shape) == observation_space.grid_shape,
        ValueError,
        'incorrect shape of visibility mask',
    )

    for pos in observation_grid.positions():
        if not visibility[pos.y, pos.x]:
            observation_grid[pos] = Hidden()

    observation_agent = Agent(
        observation_space.agent_position, Orientation.N, state.agent.obj
    )
    return Observation(observation_grid, observation_agent)


full_observation = partial(from_visibility, visibility_function=full_visibility)
"""`ObservationFunction` where every tile is visible"""

partial_observation = partial(
    from_visibility, visibility_function=partial_visibility
)
"""`ObservationFunction` which is blocked by non-transparent obstacles"""

minigrid_observation = partial(
    from_visibility, visibility_function=minigrid_visibility
)
"""`ObservationFunction` implementation as done in 'MiniGrid'"""

raytracing_observation = partial(
    from_visibility, visibility_function=raytracing_visibility
)
"""`ObservationFunction` with ray tracing"""

stochastic_raytracing_observation = partial(
    from_visibility,
    visibility_function=stochastic_raytracing_visibility,
)
"""`ObservationFunction` with stochastic ray tracing"""


def factory(
    name: str,
    *,
    observation_space: Optional[ObservationSpace] = None,
    visibility_function: Optional[VisibilityFunction] = None,
) -> ObservationFunction:

    if name == 'from_visibility':
        checkraise(
            lambda: observation_space is not None
            and visibility_function is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(
            from_visibility,
            observation_space=observation_space,
            visibility_function=visibility_function,
        )

    if name == 'full_observation':
        checkraise(
            lambda: observation_space is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(full_observation, observation_space=observation_space)

    if name == 'partial_observation':
        checkraise(
            lambda: observation_space is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(partial_observation, observation_space=observation_space)

    if name == 'minigrid_observation':
        checkraise(
            lambda: observation_space is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(
            minigrid_observation, observation_space=observation_space
        )

    if name == 'raytracing_observation':
        checkraise(
            lambda: observation_space is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(
            raytracing_observation, observation_space=observation_space
        )

    if name == 'stochastic_raytracing_observation':
        checkraise(
            lambda: observation_space is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(
            stochastic_raytracing_observation,
            observation_space=observation_space,
        )

    raise ValueError(f'invalid observation function name {name}')
