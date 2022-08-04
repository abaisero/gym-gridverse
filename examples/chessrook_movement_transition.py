from typing import Optional

import numpy.random as rnd

from gym_gridverse.action import Action
from gym_gridverse.envs.transition_functions import transition_function_registry
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.state import State

_action_orientations = {
    Action.MOVE_FORWARD: Orientation.F,
    Action.MOVE_LEFT: Orientation.L,
    Action.MOVE_RIGHT: Orientation.R,
    Action.MOVE_BACKWARD: Orientation.B,
}


@transition_function_registry.register
def chessrook_movement(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """moves the agent like a chess rook, until it hits an obstacle"""

    # get agent's movement direction
    try:
        action_orientation = _action_orientations[action]
    except KeyError:
        # not a movement action
        return

    movement_orientation = state.agent.orientation * action_orientation
    position_delta = Position.from_orientation(movement_orientation)

    # check positions until a blocking cell is found
    position = state.agent.position
    while not state.grid[position + position_delta].blocks_movement:
        position = position + position_delta

    state.agent.position = position
