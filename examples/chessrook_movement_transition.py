from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.envs.transition_functions import transition_function_registry
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
        movement_orientation = state.agent.orientation
    elif action is Action.MOVE_LEFT:
        movement_orientation = state.agent.orientation.rotate_left()
    elif action is Action.MOVE_RIGHT:
        movement_orientation = state.agent.orientation.rotate_right()
    elif action is Action.MOVE_BACKWARD:
        movement_orientation = state.agent.orientation.rotate_back()
    else:
        # not a movement action
        return

    # check positions until a blocking cell is found
    position = state.agent.position
    position_delta = movement_orientation.as_position()
    while not state.grid[position + position_delta].blocks:
        position = position + position_delta

    state.agent.position = position
