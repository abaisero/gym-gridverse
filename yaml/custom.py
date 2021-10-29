from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.agent import Agent
from gym_gridverse.design import draw_room, draw_wall_boundary
from gym_gridverse.envs.reset_functions import reset_function_registry
from gym_gridverse.envs.reward_functions import reward_function_registry
from gym_gridverse.envs.transition_functions import (
    transition_function_registry,
    update_agent,
)
from gym_gridverse.geometry import Area, Orientation
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Color, Floor, GridObject, Wall
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State


class Coin(GridObject):
    def __init__(self):
        super().__init__()
        self.state_index = 0
        self.color = Color.NONE
        self.transparent = True
        self.can_be_picked_up = False
        self.blocks = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


#  custom reset function
@reset_function_registry.register
def coin_maze(*, rng: Optional[rnd.Generator] = None) -> State:

    # must call this to include reproduceable stochasticity
    rng = get_gv_rng_if_none(rng)

    # initializes grid with Coin
    grid = Grid.from_shape((7, 9), factory=Coin)
    # assigns Wall to the border
    draw_wall_boundary(grid)
    # draw other walls
    draw_room(grid, Area((2, 4), (2, 6)), Wall)
    # re-assign openings
    grid[2, 3] = Coin()
    grid[4, 5] = Coin()

    # final result (#=Wall, .=Coin):

    # #########
    # #.......#
    # #.W.WWW.#
    # #.W...W.#
    # #.WWW.W.#
    # #.......#
    # #########

    # randomized agent position and orientation
    agent_position = rng.choice(
        [
            position
            for position in grid.area.positions()
            if isinstance(grid[position], Coin)
        ]
    )
    agent_orientation = rng.choice(list(Orientation))
    agent = Agent(agent_position, agent_orientation)

    # remove coin from agent initial position
    grid[agent.position] = Floor()

    return State(grid, agent)


#  custom transition function
@transition_function_registry.register
def multi_update_agent(
    state: State,
    action: Action,
    *,
    n: int,
    rng: Optional[rnd.Generator] = None,
):
    for _ in range(n):
        update_agent(state, action, rng=rng)


#  custom reward function
@reward_function_registry.register
def checkerboard(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_even: float,
    reward_odd: float,
    rng: Optional[rnd.Generator] = None,
):
    return (
        reward_even
        if (next_state.agent.position.y + next_state.agent.position.x) % 2 == 0
        else reward_odd
    )
