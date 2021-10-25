from typing import Type

from gym_gridverse.action import Action
from gym_gridverse.envs.observation_functions import ObservationFunction
from gym_gridverse.envs.terminating_functions import (
    terminating_function_registry,
)
from gym_gridverse.grid_object import GridObject
from gym_gridverse.state import State


@terminating_function_registry.register
def concealed_terminating(
    state: State,
    action: Action,
    next_state: State,
    *,
    object_type: Type[GridObject],
    observation_function: ObservationFunction
) -> bool:
    # NOTE:  even if the observation_function is the same, this observation
    # might be different from that received by the agent if it is stochastic.
    observation = observation_function(state)
    return any(
        isinstance(observation.grid[position], object_type)
        for position in observation.grid.area.positions()
    )
