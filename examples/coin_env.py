from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.agent import Agent
from gym_gridverse.design import draw_room, draw_wall_boundary
from gym_gridverse.envs.reset_functions import reset_function_registry
from gym_gridverse.envs.reward_functions import reward_function_registry
from gym_gridverse.envs.terminating_functions import (
    terminating_function_registry,
)
from gym_gridverse.envs.transition_functions import transition_function_registry
from gym_gridverse.geometry import Area, Orientation
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Color, Floor, GridObject, Wall
from gym_gridverse.rng import choice, get_gv_rng_if_none
from gym_gridverse.state import State


class Coin(GridObject):
    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = False
    holdable = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@reset_function_registry.register
def coin_maze(*, rng: Optional[rnd.Generator] = None) -> State:
    """creates a maze with collectible coins"""

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
    agent_position = choice(
        rng,
        [
            position
            for position in grid.area.positions()
            if isinstance(grid[position], Coin)
        ],
    )
    agent_orientation = choice(rng, list(Orientation))
    agent = Agent(agent_position, agent_orientation)

    # remove coin from agent initial position
    grid[agent.position] = Floor()

    return State(grid, agent)


@transition_function_registry.register
def collect_coin_transition(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
):
    """collects and removes coins"""
    if isinstance(state.grid[state.agent.position], Coin):
        state.grid[state.agent.position] = Floor()


@reward_function_registry.register
def collect_coin_reward(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward: float = 1.0,
    rng: Optional[rnd.Generator] = None,
):
    """gives reward if a coin was collected"""
    return (
        reward
        if isinstance(state.grid[next_state.agent.position], Coin)
        else 0.0
    )


@terminating_function_registry.register
def no_more_coins(
    state: State,
    action: Action,
    next_state: State,
    *,
    rng: Optional[rnd.Generator] = None,
):
    """terminates episodes if all coins are collected"""
    return not any(
        isinstance(next_state.grid[position], Coin)
        for position in next_state.grid.area.positions()
    )
