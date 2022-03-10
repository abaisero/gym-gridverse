from typing import Optional

import numpy.random as rnd

from gym_gridverse.agent import Agent
from gym_gridverse.envs.reset_functions import reset_function_registry
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Color, Door, Exit, Floor, Key, Wall
from gym_gridverse.rng import choice, get_gv_rng_if_none, shuffle
from gym_gridverse.state import State


@reset_function_registry.register
def choose_key(*, rng: Optional[rnd.Generator] = None) -> State:
    """the agent has to pick the correct key to open a randomly colored door"""

    rng = get_gv_rng_if_none(rng)  # necessary to use rng object!

    # only consider these colors
    colors = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]

    # randomly select key locations
    keys = shuffle(rng, [Key(color) for color in colors])
    # randomly select door color
    color = choice(rng, colors)

    # grids can be constructed directly from objects
    door = Door(Door.Status.LOCKED, color)
    grid = Grid(
        [
            [Wall(), Wall(), Wall(), Wall(), Wall()],
            [Wall(), Wall(), Exit(), Wall(), Wall()],
            [Wall(), Wall(), door, Wall(), Wall()],
            [Wall(), keys[0], Floor(), keys[1], Wall()],
            [Wall(), keys[2], Floor(), keys[3], Wall()],
            [Wall(), Wall(), Wall(), Wall(), Wall()],
        ]
    )

    # positioning the agent in the above grid
    agent = Agent(Position(4, 2), Orientation.F)

    return State(grid, agent)
