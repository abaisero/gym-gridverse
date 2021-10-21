from typing import Optional

import numpy.random as rnd

from gym_gridverse.agent import Agent
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid_object import Hidden
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import ObservationSpace
from gym_gridverse.state import State


def frontal_line_of_sight(
    state: State,
    *,
    observation_space: ObservationSpace,
    rng: Optional[rnd.Generator] = None,
) -> Observation:
    """only tiles in front of the agent can be seen"""

    # get agent's POV grid
    area = state.agent.get_pov_area(observation_space.area)
    observation_grid = state.grid.subgrid(area).change_orientation(
        state.agent.orientation
    )

    # hiding all tiles which are not directly in front of agent
    for position in observation_grid.positions():
        if (
            # the tile is not in the same column as the agent
            position.x == observation_space.agent_position.x
            # or the tile is behind the agent
            or position.y < observation_space.agent_position.y
        ):
            observation_grid[position] = Hidden()

    # boolean flag to detect whether the line of sight has run into a
    # non-transparent object
    found_non_transparent = False

    # for every y-coordinate in front of the agent
    for y in range(observation_space.agent_position.y - 1, -1, -1):
        position = Position(y, observation_space.agent_position.x)

        # hide object if we have already encountered a non-transparent object
        if found_non_transparent:
            observation_grid[position] = Hidden()

        # update the boolean flag if object is non-transparent
        found_non_transparent |= not observation_grid[position].transparent

    observation_agent = Agent(
        observation_space.agent_position, Orientation.N, state.agent.obj
    )
    return Observation(observation_grid, observation_agent)
