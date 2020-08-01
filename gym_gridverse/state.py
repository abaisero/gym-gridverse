from __future__ import annotations

import numpy as np

from .geometry import Position
from .grid_object import Hidden
from .info import Agent, Grid
from .observation import Observation


class State:
    def __init__(self, grid: Grid, agent: Agent):
        self.grid = grid
        self.agent = agent

    def observation(self) -> Observation:
        area = self.agent.get_pov_area()
        grid = self.grid.subgrid(area).change_orientation(
            self.agent.orientation
        )

        # TODO outsource this logic into some kind of ObservationFactory,
        # which handles different way that observations can be created (e.g.
        # deterministic, static)

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

        return Observation(grid, self.agent)
