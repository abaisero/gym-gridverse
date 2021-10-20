from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.grid_object import Floor, Wall
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State


def creeping_walls(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """randomly chooses a Floor tile and turns it into a Wall tile"""

    rng = get_gv_rng_if_none(rng)  # necessary to use rng object!

    # all positions associated with a Floor object
    floor_positions = [
        position
        for position in state.grid.positions()
        if isinstance(state.grid[position], Floor)
    ]

    try:
        # floor_positions could be an empty list
        position = rng.choice(floor_positions)
    except ValueError:
        # there are no floor positions
        pass
    else:
        # if we were able to sample a position, change the corresponding Floor
        # into a Wall
        state.grid[position] = Wall()
