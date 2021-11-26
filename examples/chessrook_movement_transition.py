from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.envs.transition_functions import transition_function_registry
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.state import State


@transition_function_registry.register
def chessrook_movement(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """moves the agent like a chess rook, until it hits an obstacle"""

    # get agent's movement direction
    if action is Action.MOVE_FORWARD:
        movement_orientation = state.agent.orientation * Orientation.F
    elif action is Action.MOVE_LEFT:
        movement_orientation = state.agent.orientation * Orientation.L
    elif action is Action.MOVE_RIGHT:
        movement_orientation = state.agent.orientation * Orientation.R
    elif action is Action.MOVE_BACKWARD:
        movement_orientation = state.agent.orientation * Orientation.F
    else:
        # not a movement action
        return

    # check positions until a blocking cell is found
    position = state.agent.position
    position_delta = Position.from_orientation(movement_orientation)
    while not state.grid[position + position_delta].blocks_movement:
        position = position + position_delta

    state.agent.position = position
