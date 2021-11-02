from typing import Optional

import numpy.random as rnd

from gym_gridverse.agent import Agent
from gym_gridverse.envs.reset_functions import reset_function_registry
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Exit, Floor, Wall
from gym_gridverse.state import State


@reset_function_registry.register
def simplest(*, rng: Optional[rnd.Generator] = None) -> State:
    """smallest possible room with exit right in front of agent"""

    # constructed the grid directly from objects
    grid = Grid(
        [
            [Wall(), Wall(), Wall()],
            [Wall(), Exit(), Wall()],
            [Wall(), Floor(), Wall()],
            [Wall(), Wall(), Wall()],
        ]
    )

    # positioning the agent in the above grid
    agent = Agent(Position(2, 1), Orientation.F)

    return State(grid, agent)
