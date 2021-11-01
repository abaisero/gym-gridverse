from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.envs.transition_functions import transition_function_registry
from gym_gridverse.geometry import Orientation
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
        movement_orientation = state.agent.orientation * Orientation.N
    elif action is Action.MOVE_LEFT:
        movement_orientation = state.agent.orientation * Orientation.W
    elif action is Action.MOVE_RIGHT:
        movement_orientation = state.agent.orientation * Orientation.E
    elif action is Action.MOVE_BACKWARD:
        movement_orientation = state.agent.orientation * Orientation.N
    else:
        # not a movement action
        return

    # check positions until a blocking cell is found
    position = state.agent.position
    position_delta = movement_orientation.as_position()
    while not state.grid[position + position_delta].blocks:
        position = position + position_delta

    state.agent.position = position
