import inspect
import warnings
from functools import partial
from typing import Callable, Iterator, List, Optional, Sequence, Type

import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.action import Action
from gym_gridverse.envs.utils import get_next_position
from gym_gridverse.grid_object import Exit, GridObject, MovingObstacle, Wall
from gym_gridverse.state import State
from gym_gridverse.utils.custom import import_if_custom
from gym_gridverse.utils.functions import checkraise_kwargs, select_kwargs
from gym_gridverse.utils.protocols import (
    get_keyword_parameter,
    get_positional_parameters,
)
from gym_gridverse.utils.registry import FunctionRegistry


class TerminatingFunction(Protocol):
    """Signature for functions to determine whether a transition is terminal"""

    def __call__(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> bool:
        ...


TerminatingReductionFunction = Callable[[Iterator[bool]], bool]
"""Signature for a boolean reduction function"""


class TerminatingFunctionRegistry(FunctionRegistry):
    def get_protocol_parameters(
        self, signature: inspect.Signature
    ) -> List[inspect.Parameter]:
        state, action, next_state = get_positional_parameters(signature, 3)
        rng = get_keyword_parameter(signature, 'rng')
        return [state, action, next_state, rng]

    def check_signature(self, function: TerminatingFunction):
        signature = inspect.signature(function)
        state, action, next_state, rng = self.get_protocol_parameters(signature)

        # checks first 3 arguments are positional
        if state.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise TypeError(
                f'The first argument ({state.name}) '
                f'of a registered terminating function ({function}) '
                'should be allowed to be a positional argument.'
            )

        if action.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise TypeError(
                f'The second argument ({action.name}) '
                f'of a registered terminating function ({function}) '
                'should be allowed to be a positional argument.'
            )

        if next_state.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise TypeError(
                f'The third argument ({next_state.name}) '
                f'of a registered terminating function ({function}) '
                'should be allowed to be a positional argument.'
            )

        # and `rng` is keyword
        if rng.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ]:
            raise TypeError(
                f'The `rng` argument ({rng.name}) '
                f'of a registered reward function ({function}) '
                'should be allowed to be a keyword argument.'
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

        if rng.annotation not in [
            inspect.Parameter.empty,
            Optional[rnd.Generator],
        ]:
            warnings.warn(
                f'The `rng` argument ({rng.name}) '
                f'of a registered reward function ({function}) '
                f'has an annotation ({rng.annotation}) '
                'which is not `Optional[rnd.Generator]`.'
            )

        if signature.return_annotation not in [inspect.Parameter.empty, bool]:
            warnings.warn(
                f'The return type of a registered terminating function ({function}) '
                f'has an annotation ({signature.return_annotation}) '
                'which is not `bool`.'
            )


terminating_function_registry = TerminatingFunctionRegistry()
"""Terminating function registry"""


@terminating_function_registry.register
def reduce(
    state: State,
    action: Action,
    next_state: State,
    *,
    terminating_functions: Sequence[TerminatingFunction],
    reduction: TerminatingReductionFunction,
    rng: Optional[rnd.Generator] = None,
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
        terminating_function(state, action, next_state, rng=rng)
        for terminating_function in terminating_functions
    )


@terminating_function_registry.register
def reduce_any(
    state: State,
    action: Action,
    next_state: State,
    *,
    terminating_functions: Sequence[TerminatingFunction],
    rng: Optional[rnd.Generator] = None,
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
        rng=rng,
    )


@terminating_function_registry.register
def reduce_all(
    state: State,
    action: Action,
    next_state: State,
    *,
    terminating_functions: Sequence[TerminatingFunction],
    rng: Optional[rnd.Generator] = None,
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
        rng=rng,
    )


@terminating_function_registry.register
def overlap(
    state: State,
    action: Action,
    next_state: State,
    *,
    object_type: Type[GridObject],
    rng: Optional[rnd.Generator] = None,
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
def reach_exit(
    state: State,
    action: Action,
    next_state: State,
    *,
    rng: Optional[rnd.Generator] = None,
) -> bool:
    """terminating condition for Agent reaching the Exit

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):

    Returns:
        bool: True if next_state agent is on exit
    """
    return overlap(state, action, next_state, object_type=Exit, rng=rng)


@terminating_function_registry.register
def bump_moving_obstacle(
    state: State,
    action: Action,
    next_state: State,
    *,
    rng: Optional[rnd.Generator] = None,
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
    return overlap(
        state, action, next_state, object_type=MovingObstacle, rng=rng
    )


@terminating_function_registry.register
def bump_into_wall(
    state: State,
    action: Action,
    next_state: State,
    *,
    rng: Optional[rnd.Generator] = None,
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
    next_position = get_next_position(
        state.agent.position, state.agent.orientation, action
    )

    return state.grid.area.contains(next_position) and isinstance(
        state.grid[next_position], Wall
    )


def factory(name: str, **kwargs) -> TerminatingFunction:
    name = import_if_custom(name)

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
