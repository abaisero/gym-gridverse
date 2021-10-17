from functools import partial
from typing import Callable, Dict, Iterator, List, Sequence, Type

from gym_gridverse.action import Action
from gym_gridverse.envs.utils import updated_agent_position_if_unobstructed
from gym_gridverse.grid_object import Exit, GridObject, MovingObstacle, Wall
from gym_gridverse.state import State
from gym_gridverse.utils.functions import (
    checkraise_kwargs,
    get_custom_function,
    is_custom_function,
    select_kwargs,
)

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
    # TODO: test
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
    # TODO: test
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
    # TODO: test
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


def reach_exit(state: State, action: Action, next_state: State) -> bool:
    """terminating condition for Agent reaching the Exit

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):

    Returns:
        bool: True if next_state agent is on exit
    """
    return overlap(state, action, next_state, object_type=Exit)


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
    # TODO: test
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


def factory(name: str, **kwargs) -> TerminatingFunction:

    required_keys: List[str]
    optional_keys: List[str]

    if name == 'reduce':
        required_keys = ['terminating_functions', 'reduction']
        optional_keys = []
        checkraise_kwargs(kwargs, required_keys)
        kwargs = select_kwargs(kwargs, required_keys + optional_keys)
        return partial(reduce_any, **kwargs)

    if name == 'reduce_any':
        required_keys = ['terminating_functions']
        optional_keys = []
        checkraise_kwargs(kwargs, required_keys)
        kwargs = select_kwargs(kwargs, required_keys + optional_keys)
        return partial(reduce_any, **kwargs)

    if name == 'reduce_all':
        required_keys = ['terminating_functions']
        optional_keys = []
        checkraise_kwargs(kwargs, required_keys)
        kwargs = select_kwargs(kwargs, required_keys + optional_keys)
        return partial(reduce_all, **kwargs)

    if name == 'overlap':
        required_keys = ['object_type']
        optional_keys = []
        checkraise_kwargs(kwargs, required_keys)
        kwargs = select_kwargs(kwargs, required_keys + optional_keys)
        return partial(overlap, **kwargs)

    if name == 'reach_exit':
        return reach_exit

    if name == 'bump_moving_obstacle':
        return bump_moving_obstacle

    if name == 'bump_into_wall':
        return bump_into_wall

    if is_custom_function(name):
        function = get_custom_function(name)
        return partial(function, **kwargs)

    raise ValueError(f'invalid terminating function name `{name}`')
