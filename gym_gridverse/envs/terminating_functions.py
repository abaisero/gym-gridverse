from typing import Callable

from gym_gridverse.envs import Actions
from gym_gridverse.grid_object import Goal
from gym_gridverse.state import State

TerminatingFunction = Callable[[State, Actions, State], bool]


def reach_goal(
    state: State,  # pylint: disable=unused-argument
    action: Actions,  # pylint: disable=unused-argument
    next_state: State,
) -> bool:
    """terminating condition for Agent reaching the Goal

    Args:
        state (`State`):
        action (`Actions`):
        next_state (`State`):

    Returns:
        bool: True if next_state agent is on goal
    """
    return isinstance(next_state.grid[next_state.agent.position], Goal)
