from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.envs.transition_functions import transition_function_registry
from gym_gridverse.grid_object import Floor, Wall
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State


@transition_function_registry.register
def creeping_walls(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """randomly converts a Floor into a Wall"""

    rng = get_gv_rng_if_none(rng)  # necessary to use rng object!

    # all Floor positions
    floor_positions = [
        position
        for position in state.grid.positions()
        if isinstance(state.grid[position], Floor)
    ]

    try:
        # choose a random Floor position ...
        position = rng.choice(floor_positions)
    except ValueError:
        # (floor_positions was an empty list)
        pass
    else:
        # ... and turn it into a Wall
        state.grid[position] = Wall()