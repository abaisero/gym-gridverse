""" Functions to model dynamics """
import inspect
import warnings
from functools import partial
from typing import List, Optional, Sequence

import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.action import Action
from gym_gridverse.envs.utils import get_next_position
from gym_gridverse.geometry import Orientation, get_manhattan_boundary
from gym_gridverse.grid_object import (
    Box,
    Door,
    Floor,
    Key,
    MovingObstacle,
    NoneGridObject,
    Telepod,
)
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State
from gym_gridverse.utils.custom import import_if_custom
from gym_gridverse.utils.fast_copy import fast_copy
from gym_gridverse.utils.functions import checkraise_kwargs, select_kwargs
from gym_gridverse.utils.protocols import (
    get_keyword_parameter,
    get_positional_parameters,
)
from gym_gridverse.utils.registry import FunctionRegistry


class TransitionFunction(Protocol):
    """Signature that all reset functions must follow"""

    def __call__(
        self,
        state: State,
        action: Action,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        ...


class TransitionFunctionRegistry(FunctionRegistry):
    def get_protocol_parameters(
        self, signature: inspect.Signature
    ) -> List[inspect.Parameter]:
        state, action = get_positional_parameters(signature, 2)
        rng = get_keyword_parameter(signature, 'rng')
        return [state, action, rng]

    def check_signature(self, function: TransitionFunction):
        signature = inspect.signature(function)
        state, action, rng = self.get_protocol_parameters(signature)

        # checks first 2 arguments are positional ...
        if state.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise TypeError(
                f'The first argument ({state.name}) '
                f'of a registered reward function ({function}) '
                'should be allowed to be a positional argument.'
            )

        if action.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise TypeError(
                f'The second argument ({action.name}) '
                f'of a registered reward function ({function}) '
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
                f'of a registered reward function ({function}) '
                f'has an annotation ({state.annotation}) '
                'which is not `State`.'
            )

        if action.annotation not in [inspect.Parameter.empty, Action]:
            warnings.warn(
                f'The second argument ({action.name}) '
                f'of a registered reward function ({function}) '
                f'has an annotation ({action.annotation}) '
                'which is not `Action`.'
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

        if signature.return_annotation not in [inspect.Parameter.empty, None]:
            warnings.warn(
                f'The return type of a registered transition function ({function}) '
                f'has an annotation ({signature.return_annotation}) '
                'which is not `None`.'
            )


transition_function_registry = TransitionFunctionRegistry()
"""Transition function registry"""


@transition_function_registry.register
def chain(
    state: State,
    action: Action,
    *,
    transition_functions: Sequence[TransitionFunction],
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Run multiple transition functions in a row

    Args:
        state (`State`):
        action (`Action`):
        transition_functions (`Sequence[TransitionFunction]`): transition functions
        rng (`Generator, optional`)

    Returns:
        None
    """
    for transition_function in transition_functions:
        transition_function(state, action, rng=rng)


@transition_function_registry.register
def move_agent(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Applies translation to agent (e.g. up/down/left/right)

    Leaves the state unaffected if any other action was taken instead

    Args:
        state (`State`):
        action (`Action`):

    Returns:
        None:
    """

    if not action.is_move():
        return

    next_position = get_next_position(
        state.agent.position,
        state.agent.orientation,
        action,
    )

    try:
        obj = state.grid[next_position]
    except IndexError:
        pass
    else:
        if not obj.blocks_movement:
            state.agent.position = next_position


@transition_function_registry.register
def turn_agent(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Turns agent according to action (e.g. turn left/right)

    Leaves the state unaffected if any other action was taken instead

    Args:
        state (`State`):
        action (`Action`):

    Returns:
        None:
    """

    try:
        orientation = _action_orientations[action]
    except KeyError:
        pass
    else:
        state.agent.orientation *= orientation


# for turn_agent
_action_orientations = {
    Action.TURN_LEFT: Orientation.L,
    Action.TURN_RIGHT: Orientation.R,
}


@transition_function_registry.register
def pickndrop(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Implements the effect of the pickup and drop action

    Pickup applies to the item *in front* of the agent
    There are multiple scenarios

    * There is no (pick-up-able) item to pickup under the agent:
        * The agent is not holding any object -> No effect
        * The agent is holding an object:
            * Position in front of agent is floor -> drop current object
            * Position in front is not a floor -> No effect
    * There is a (pick-up-able) item to pickup under the agent:
        * The agent is not holding any object -> Pick up, put floor in stead
        * The agent is holding an object -> Swap items

    Args:
        state (`State`):
        action (`Action`):
        rng (`Generator, optional`)

    Returns:
        None:
    """

    if action is not Action.PICK_N_DROP:
        return

    position_front = state.agent.front()
    obj_front = state.grid[position_front]
    can_be_dropped = isinstance(obj_front, Floor) or obj_front.holdable

    if not can_be_dropped:
        return

    state.grid[position_front] = (
        state.agent.grid_object
        if not isinstance(state.agent.grid_object, NoneGridObject)
        and can_be_dropped
        else Floor()  # We know we are picking up if not dropping
    )

    state.agent.grid_object = (
        obj_front if obj_front.holdable else NoneGridObject()
    )


@transition_function_registry.register
def move_obstacles(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Moves moving obstacles randomly

    Randomly moves each MovingObstacle to a neighbouring Floor cell, if possible.

    Args:
        state (`State`): current state
        action (`Action`): action taken by agent (ignored)
    """
    rng = get_gv_rng_if_none(rng)

    # get all positions before performing any movement
    positions = [
        position
        for position in state.grid.area.positions()
        if isinstance(state.grid[position], MovingObstacle)
    ]

    for position in positions:
        next_positions = [
            next_position
            for next_position in get_manhattan_boundary(position, distance=1)
            if state.grid.area.contains(next_position)
            and isinstance(state.grid[next_position], Floor)
        ]

        try:
            i = rng.choice(len(next_positions))
        except ValueError:
            pass
        else:
            next_position = next_positions[i]
            state.grid.swap(position, next_position)


@transition_function_registry.register
def actuate_door(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Attempts to open door

    When not holding correct key with correct color:
        `open` or `closed` -> `open`
        `locked` -> `locked`

    When holding correct key:
        any state -> `open`

    """

    if action is not Action.ACTUATE:
        return

    position = state.agent.front()

    if not state.grid.area.contains(position):
        return

    door = state.grid[position]

    if not isinstance(door, Door):
        return

    if door.is_open:
        pass

    elif not door.is_locked:
        door.state = Door.Status.OPEN

    else:
        if (
            isinstance(state.agent.grid_object, Key)
            and state.agent.grid_object.color == door.color
        ):
            door.state = Door.Status.OPEN


@transition_function_registry.register
def actuate_box(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Attempts to open door

    When not holding correct key with correct color:
        `open` or `closed` -> `open`
        `locked` -> `locked`

    When holding correct key:
        any state -> `open`

    """

    if action is not Action.ACTUATE:
        return

    position = state.agent.front()

    if not state.grid.area.contains(position):
        return

    box = state.grid[position]

    if isinstance(box, Box):
        state.grid[position] = box.content


@transition_function_registry.register
def teleport(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Teleports the agent if positioned on the telepod"""

    rng = get_gv_rng_if_none(rng)

    telepod = state.grid[state.agent.position]

    if isinstance(telepod, Telepod):
        positions = [
            position
            for position in state.grid.area.positions()
            if position != state.agent.position
            and isinstance(state.grid[position], Telepod)
            and state.grid[position].color == telepod.color
        ]
        i = rng.choice(len(positions))
        state.agent.position = positions[i]


def factory(name: str, **kwargs) -> TransitionFunction:
    name = import_if_custom(name)

    try:
        function = transition_function_registry[name]
    except KeyError as error:
        raise ValueError(f'invalid transition function name {name}') from error

    signature = inspect.signature(function)
    required_keys = [
        parameter.name
        for parameter in transition_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is inspect.Parameter.empty
    ]
    optional_keys = [
        parameter.name
        for parameter in transition_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is not inspect.Parameter.empty
    ]

    checkraise_kwargs(kwargs, required_keys)
    kwargs = select_kwargs(kwargs, required_keys + optional_keys)
    return partial(function, **kwargs)


def transition_with_copy(
    transition_function: TransitionFunction,
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    """Utility to perform a non-in-place version of a transition function.

    NOTE:  This is *not* a transition function (transition functions are
    in-place by definition).

    Args:
        transition_function (`TransitionFunction`):
        state (`State`):
        action (`action`):
        rng (`Generator, optional`)

    Returns:
        State:
    """
    next_state = fast_copy(state)
    transition_function(next_state, action, rng=rng)
    return next_state
