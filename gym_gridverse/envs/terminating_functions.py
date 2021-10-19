import inspect
import itertools as itt
import warnings
from functools import partial
from typing import Callable, Iterator, List, Sequence, Type

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
from gym_gridverse.utils.registry import FunctionRegistry

TerminatingFunction = Callable[[State, Action, State], bool]
"""Signature for functions to determine whether a transition is terminal"""

TerminatingReductionFunction = Callable[[Iterator[bool]], bool]
"""Signature for a boolean reduction function"""


class TerminatingFunctionRegistry(FunctionRegistry):
    def get_protocol_parameters(
        self, signature: inspect.Signature
    ) -> List[inspect.Parameter]:
        state, action, next_state = itt.islice(signature.parameters.values(), 3)
        return [state, action, next_state]

    def check_signature(self, function: TerminatingFunction):
        signature = inspect.signature(function)
        state, action, next_state = self.get_protocol_parameters(signature)

        # checks first 3 arguments are positional
        if state.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise ValueError(
                f'The first argument ({state.name}) '
                f'of a registered terminating function ({function}) '
                'should be allowed to be a positional argument.'
            )

        if action.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise ValueError(
                f'The second argument ({action.name}) '
                f'of a registered terminating function ({function}) '
                'should be allowed to be a positional argument.'
            )

        if next_state.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise ValueError(
                f'The third argument ({next_state.name}) '
                f'of a registered terminating function ({function}) '
                'should be allowed to be a positional argument.'
            )

        # checks if annotations, if given, are consistent
        if state.annotation not in [inspect.Parameter.empty, State]:
            warnings.warn(
                f'The first argument ({state.name}) '
                f'of a registered terminating function ({function}) '
                f'has an annotation ({state.annotation}) '
                'which is not `State`.'
            )

        if action.annotation not in [inspect.Parameter.empty, Action]:
            warnings.warn(
                f'The second argument ({action.name}) '
                f'of a registered terminating function ({function}) '
                f'has an annotation ({action.annotation}) '
                'which is not `Action`.'
            )

        if next_state.annotation not in [inspect.Parameter.empty, State]:
            warnings.warn(
                f'The third argument ({next_state.name}) '
                f'of a registered terminating function ({function}) '
                f'has an annotation ({next_state.annotation}) '
                'which is not `State`.'
            )

        if signature.return_annotation not in [inspect.Parameter.empty, bool]:
            warnings.warn(
                f'The return type of a registered terminating function ({function}) '
                f'has an annotation ({signature.return_annotation}) '
                'which is not `bool`.'
            )


terminating_function_registry = TerminatingFunctionRegistry()


@terminating_function_registry.register
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


@terminating_function_registry.register
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


@terminating_function_registry.register
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


@terminating_function_registry.register
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


@terminating_function_registry.register
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


@terminating_function_registry.register
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


@terminating_function_registry.register
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

    if is_custom_function(name):
        name = import_custom_function(name)

    try:
        function = terminating_function_registry[name]
    except KeyError as error:
        raise ValueError(f'invalid terminating function name {name}') from error

    signature = inspect.signature(function)
    required_keys = [
        parameter.name
        for parameter in terminating_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is inspect.Parameter.empty
    ]
    optional_keys = [
        parameter.name
        for parameter in terminating_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is not inspect.Parameter.empty
    ]

    checkraise_kwargs(kwargs, required_keys)
    kwargs = select_kwargs(kwargs, required_keys + optional_keys)
    return partial(function, **kwargs)
