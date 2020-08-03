from typing import Callable

import numpy as np

from gym_gridverse.geometry import Position
from gym_gridverse.grid_object import Hidden
from gym_gridverse.observation import Observation
from gym_gridverse.state import State

ObservationFunction = Callable[[State], Observation]


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

    return Observation(grid, state.agent)
