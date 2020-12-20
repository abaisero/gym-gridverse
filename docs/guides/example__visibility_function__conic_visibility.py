from typing import Optional

import numpy as np
import numpy.random as rnd

from gym_gridverse.geometry import Position
from gym_gridverse.grid import Grid


def conic_visibility(
    grid: Grid,
    position: Position,
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
) -> np.ndarray:
    """cone-shaped visibility, passes through objects"""

    # initialize visibility matrix to False
    visibility = np.zeros(grid.shape, dtype=bool)

    # for every y-coordinate in front of the agent
    for y in range(position.y, -1, -1):
        dy = position.y - y

        x_from = position.x - dy
        x_to = position.x + dy

        # for every x-coordinate within the cone range
        for x in range(x_from, x_to + 1):
            # tile in the cone is visible
            visibility[y, x] = True

    return visibility
