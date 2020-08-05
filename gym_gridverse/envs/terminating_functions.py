from typing import Callable, Sequence, Type

from gym_gridverse.actions import Actions
from gym_gridverse.grid_object import Goal, GridObject, MovingObstacle
from gym_gridverse.state import State

TerminatingFunction = Callable[[State, Actions, State], bool]


def chain_any(
    state: State,
    action: Actions,
    next_state: State,
    *,
    terminating_functions: Sequence[TerminatingFunction],
) -> bool:
    """utility function terminates when any of the input functions terminates

    Args:
        state (`State`):
        action (`Actions`):
        next_state (`State`):
        terminating_functions (`Sequence[TerminatingFunction]`):

    Returns:
        bool: OR operator over the input terminating functions
    """
    return any(
        terminating_function(state, action, next_state)
        for terminating_function in terminating_functions
    )


def chain_all(
    state: State,
    action: Actions,
    next_state: State,
    *,
    terminating_functions: Sequence[TerminatingFunction],
) -> bool:
    """utility function terminates when all of the input functions terminates

    Args:
        state (`State`):
        action (`Actions`):
        next_state (`State`):
        terminating_functions (`Sequence[TerminatingFunction]`):

    Returns:
        bool: AND operator over the input terminating functions
    """
    return all(
        terminating_function(state, action, next_state)
        for terminating_function in terminating_functions
    )


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


def bump_moving_obstacle(
    state: State, action: Actions, next_state: State,
) -> bool:
    """terminating condition for Agent bumping a moving obstacle

    Args:
        state (`State`):
        action (`Actions`):
        next_state (`State`):

    Returns:
        bool: True if next_state agent is on a MovingObstacle
    """
    return overlap(state, action, next_state, object_type=MovingObstacle)
