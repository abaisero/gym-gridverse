from typing import Optional

import numpy.random as rnd

from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import Color, Door, Floor, Goal, Key, Wall
from gym_gridverse.info import Agent, Grid
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State


def match_key_color(
    *,
    rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
) -> State:
    """the agent has to pick the correct key to open a randomly colored door"""

    rng = get_gv_rng_if_none(rng)  # necessary to use rng object!

    # only consider these colors
    colors = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]
    # randomly choose location of keys
    key1, key2, key3, key4 = rng.permute([Key(color) for color in colors])
    # randomly choose color of door
    door = Door(Door.Status.LOCKED, rng.choice(colors))

    # grids can be constructed directly from objects
    grid = Grid.from_objects(
        [
            [Wall(), Wall(), Wall(), Wall(), Wall()],
            [Wall(), Wall(), Goal(), Wall(), Wall()],
            [Wall(), Wall(), door, Wall(), Wall()],
            [Wall(), key1, Floor(), key2, Wall()],
            [Wall(), key3, Floor(), key4, Wall()],
            [Wall(), Wall(), Wall(), Wall(), Wall()],
        ]
    )

    # positioning the agent in the above grid
    agent = Agent((4, 2), Orientation.N)

    return State(grid, agent)
