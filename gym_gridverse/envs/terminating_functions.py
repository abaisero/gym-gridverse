from functools import partial
from typing import Callable, Iterator, Optional, Sequence, Type

from gym_gridverse.action import Action
from gym_gridverse.debugging import checkraise
from gym_gridverse.envs.utils import updated_agent_position_if_unobstructed
from gym_gridverse.grid_object import Goal, GridObject, MovingObstacle, Wall
from gym_gridverse.state import State

TerminatingFunction = Callable[[State, Action, State], bool]
"""Signature for functions to determine whether a transition is terminal"""


TerminatingReductionFunction = Callable[[Iterator[bool]], bool]
"""Signature for a boolean reduction function"""


def reduce(
    state: State,
    action: Action,
    next_state: State,
    *,
    terminating_functions: Sequence[TerminatingFunction],
    reduction: TerminatingReductionFunction,
) -> bool:
    """reduction of multiple terminating functions into a single boolean value

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        terminating_functions (`Sequence[TerminatingFunction]`):
        reduction (`TerminatingReductionFunction`):

    Returns:
        bool: reduction operator over the input terminating functions
    """
    return reduction(
        terminating_function(state, action, next_state)
        for terminating_function in terminating_functions
    )


def reduce_any(
    state: State,
    action: Action,
    next_state: State,
    *,
    terminating_functions: Sequence[TerminatingFunction],
) -> bool:
    """utility function terminates when any of the input functions terminates

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        terminating_functions (`Sequence[TerminatingFunction]`):

    Returns:
        bool: OR operator over the input terminating functions
    """
    return reduce(
        state,
        action,
        next_state,
        terminating_functions=terminating_functions,
        reduction=any,
    )


def reduce_all(
    state: State,
    action: Action,
    next_state: State,
    *,
    terminating_functions: Sequence[TerminatingFunction],
) -> bool:
    """utility function terminates when all of the input functions terminates

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        terminating_functions (`Sequence[TerminatingFunction]`):

    Returns:
        bool: AND operator over the input terminating functions
    """
    return reduce(
        state,
        action,
        next_state,
        terminating_functions=terminating_functions,
        reduction=all,
    )


def overlap(
    state: State,  # pylint: disable=unused-argument
    action: Action,  # pylint: disable=unused-argument
    next_state: State,
    *,
    object_type: Type[GridObject],
) -> bool:
    """terminating condition for agent occupying same position as an object

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        object_type (`Type[GridObject]`):

    Returns:
        bool: True if next_state agent is on object of type object_type
    """
    return isinstance(next_state.grid[next_state.agent.position], object_type)


def reach_goal(state: State, action: Action, next_state: State) -> bool:
    """terminating condition for Agent reaching the Goal

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):

    Returns:
        bool: True if next_state agent is on goal
    """
    return overlap(state, action, next_state, object_type=Goal)


def bump_moving_obstacle(
    state: State, action: Action, next_state: State
) -> bool:
    """terminating condition for Agent bumping a moving obstacle

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):

    Returns:
        bool: True if next_state agent is on a MovingObstacle
    """
    return overlap(state, action, next_state, object_type=MovingObstacle)


def bump_into_wall(
    state: State,
    action: Action,
    next_state: State,  # pylint: disable=unused-argument
) -> bool:
    """Terminating condition for Agent bumping into a wall

    Tests whether the intended next agent position from state contains a Wall

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):

    Returns:
        bool: True if next_state agent attempted to move onto a wall cell
    """
    attempted_next_position = updated_agent_position_if_unobstructed(
        state.agent.position, state.agent.orientation, action
    )

    return attempted_next_position in state.grid and (
        isinstance(state.grid[attempted_next_position], Wall)
    )


def factory(
    name: str,
    *,
    terminating_functions: Optional[Sequence[TerminatingFunction]] = None,
    reduction: Optional[TerminatingReductionFunction] = None,
    object_type: Optional[Type[GridObject]] = None,
) -> TerminatingFunction:

    if name == 'reduce':
        checkraise(
            lambda: terminating_functions is not None and reduction is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(
            reduce_any,
            terminating_functions=terminating_functions,
            reduction=reduction,
        )

    if name == 'reduce_any':
        checkraise(
            lambda: terminating_functions is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(reduce_any, terminating_functions=terminating_functions)

    if name == 'reduce_all':
        checkraise(
            lambda: terminating_functions is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(reduce_all, terminating_functions=terminating_functions)

    if name == 'overlap':
        checkraise(
            lambda: object_type is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(overlap, object_type=object_type)

    if name == 'reach_goal':
        return reach_goal

    if name == 'bump_moving_obstacle':
        return bump_moving_obstacle

    if name == 'bump_into_wall':
        return bump_into_wall

    raise ValueError(f'invalid terminating function name `{name}`')
