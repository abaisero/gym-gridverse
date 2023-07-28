from typing import Optional

import numpy.random as rnd

from gym_gridverse.agent import Agent
from gym_gridverse.envs.observation_functions import (
    observation_function_registry,
)
from gym_gridverse.geometry import Area, Position, Shape
from gym_gridverse.observation import Observation
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State


@observation_function_registry.register
def satellite(
    state: State,
    *,
    shape: Shape,
    rng: Optional[rnd.Generator] = None,
) -> Observation:
    rng = get_gv_rng_if_none(rng)  # necessary to use rng object!

    # randomly sample an area of the given shape which includes the agent
    # (also avoids putting the agent on the edge)
    y0: int = rng.integers(
        state.agent.position.y - shape.height + 2,
        state.agent.position.y - 1,
        endpoint=True,
    )
    x0: int = rng.integers(
        state.agent.position.x - shape.width + 2,
        state.agent.position.x - 1,
        endpoint=True,
    )
    area = Area((y0, y0 + shape.height - 1), (x0, x0 + shape.width - 1))
    observation_grid = state.grid.subgrid(area)

    observation_agent = Agent(
        state.agent.position - Position(y0, x0),
        state.agent.orientation,
        state.agent.grid_object,
    )

    return Observation(observation_grid, observation_agent)
