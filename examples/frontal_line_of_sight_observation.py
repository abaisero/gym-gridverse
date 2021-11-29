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
    area = state.agent.transform * observation_space.area
    observation_grid = state.grid.subgrid(area) * state.agent.orientation

    # hiding all tiles which are not directly in front of agent
    for position in observation_grid.area.positions():
        if (
            # the tile is not in the same column as the agent
            position.x == observation_space.agent_position.x
            # or the tile is behind the agent
            or position.y < observation_space.agent_position.y
        ):
            observation_grid[position] = Hidden()

    # boolean flag to detect whether the line of sight has run into a
    # vision-blocking object
    found_blocks_vision = False

    # for every y-coordinate in front of the agent
    for y in range(observation_space.agent_position.y - 1, -1, -1):
        position = Position(y, observation_space.agent_position.x)

        # hide object if we have already encountered a vision-blocking object
        if found_blocks_vision:
            observation_grid[position] = Hidden()

        # update the boolean flag if object blocks vision
        found_blocks_vision |= observation_grid[position].blocks_vision

    observation_agent = Agent(
        observation_space.agent_position, Orientation.F, state.agent.grid_object
    )
    return Observation(observation_grid, observation_agent)
