from typing import Callable, Type

from gym_gridverse.envs import Actions
from gym_gridverse.grid_object import Goal, GridObject
from gym_gridverse.state import State

TerminatingFunction = Callable[[State, Actions, State], bool]


def overlap(
    state: State,  # pylint: disable=unused-argument
    action: Actions,  # pylint: disable=unused-argument
    next_state: State,
    *,
    object_type: Type[GridObject],
) -> bool:
    """terminating condition for agent occupying same position as an object

    Args:
        state (`State`):
        action (`Actions`):
        next_state (`State`):
        object_type (`Type[GridObject]`):

    Returns:
        bool: True if next_state agent is on object of type object_type
    """
    return isinstance(next_state.grid[next_state.agent.position], object_type)


def reach_goal(state: State, action: Actions, next_state: State,) -> bool:
    """terminating condition for Agent reaching the Goal

    Args:
        state (`State`):
        action (`Actions`):
        next_state (`State`):

    Returns:
        bool: True if next_state agent is on goal
    """
    return overlap(state, action, next_state, object_type=Goal)
