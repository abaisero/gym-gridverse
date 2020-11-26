from typing import Optional

import numpy.random as rnd

from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import Floor, Goal, Wall
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State


def simplest_reset(
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
) -> State:
    """smallest possible room with goal right in front of agent"""

    # constructed the grid directly from objects
    grid = Grid.from_objects(
        [
            [Wall(), Wall(), Wall()],
            [Wall(), Goal(), Wall()],
            [Wall(), Floor(), Wall()],
            [Wall(), Wall(), Wall()],
        ]
    )

    # positioning the agent in the above grid
    agent = Agent((2, 1), Orientation.N)

    return State(grid, agent)
