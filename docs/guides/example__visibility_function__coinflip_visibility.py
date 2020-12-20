from typing import Optional

import numpy as np
import numpy.random as rnd

from gym_gridverse.geometry import Position
from gym_gridverse.grid import Grid
from gym_gridverse.rng import get_gv_rng_if_none


def coinflip_visibility(
    grid: Grid,
    position: Position,
    *,
    rng: Optional[rnd.Generator] = None,
) -> np.ndarray:
    """randomly determines tile visibility"""

    rng = get_gv_rng_if_none(rng)  # necessary to use rng object!

    # sample a random binary visibility matrix
    visibility = rng.integers(2, size=grid.shape).astype(bool)

    # the agent tile should always be visible regardless
    visibility[position.y, position.x] = True

    return visibility
