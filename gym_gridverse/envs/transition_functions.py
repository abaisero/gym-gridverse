""" Functions to model dynamics """
import inspect
import warnings
from functools import partial
from typing import Iterator, List, Optional, Sequence, Tuple, Type

import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.action import ROTATION_ACTIONS, TRANSLATION_ACTIONS, Action
from gym_gridverse.agent import Agent
from gym_gridverse.envs.utils import updated_agent_position_if_unobstructed
from gym_gridverse.geometry import Position, get_manhattan_boundary
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import (
    Box,
    Door,
    Floor,
    GridObject,
    Key,
    MovingObstacle,
    NoneGridObject,
    Telepod,
)
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State
from gym_gridverse.utils.functions import (
    checkraise_kwargs,
    import_custom_function,
    is_custom_function,
    select_kwargs,
)
from gym_gridverse.utils.protocols import (
    get_keyword_parameter,
    get_positional_parameters,
)
from gym_gridverse.utils.registry import FunctionRegistry


class TransitionFunction(Protocol):
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


# TODO: move these non-transition functions elsewhere; they are confusing


def move_agent(agent: Agent, grid: Grid, action: Action) -> None:
    """Applies translation to agent (e.g. up/down/left/right)

    Leaves the state unaffected if any other action was taken instead

    Args:
        agent (`Agent`):
        grid (`Grid`):
        action (`Action`):

    Returns:
        None:
    """

    if action not in TRANSLATION_ACTIONS:
        return

    next_pos = updated_agent_position_if_unobstructed(
        agent.position, agent.orientation, action
    )

    # exit if next position is not legal
    if not grid.area.contains(next_pos) or grid[next_pos].blocks:
        return

    agent.position = next_pos


def rotate_agent(agent: Agent, action: Action) -> None:
    """Rotates agent according to action (e.g. turn left/right)

    Leaves the state unaffected if any other action was taken instead

    Args:
        agent (`Agent`):
        action (`Action`):

    Returns:
        None:
    """

    if action not in ROTATION_ACTIONS:
        return

    if action == Action.TURN_LEFT:
        agent.orientation = agent.orientation.rotate_left()

    if action == Action.TURN_RIGHT:
        agent.orientation = agent.orientation.rotate_right()


@transition_function_registry.register
def update_agent(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Simply updates the agents location and orientation based on action

    If action does not affect this (e.g. not turning not moving), then leaves
    the state untouched

    Args:
        state (`State`):
        action (`Action`):
        rng (`Generator, optional`)

    Returns:
        None
    """
    if action not in TRANSLATION_ACTIONS and action not in ROTATION_ACTIONS:
        return

    if action in ROTATION_ACTIONS:
        rotate_agent(state.agent, action)

    if action in TRANSLATION_ACTIONS:
        move_agent(state.agent, state.grid, action)


@transition_function_registry.register
def pickup_mechanics(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Implements the effect of the pickup and drop action

    Pickup applies to the item *in front* of the agent
    There are multiple scenarii

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

    if action != Action.PICK_N_DROP:
        return

    obj_in_front_of_agent = state.grid[state.agent.position_in_front()]
    obj_holding = state.agent.obj

    can_pickup = obj_in_front_of_agent.can_be_picked_up
    can_drop = isinstance(obj_in_front_of_agent, Floor) or can_pickup

    if not can_pickup and not can_drop:
        return

    state.grid[state.agent.position_in_front()] = (
        obj_holding
        if can_drop and not isinstance(obj_holding, NoneGridObject)
        else Floor()  # We know we are picking up if not dropping
    )

    # Know for sure that if not can_pickup then we have dropped
    state.agent.obj = obj_in_front_of_agent if can_pickup else NoneGridObject()


def _unique_object_type_positions(
    grid: Grid, object_type: Type[GridObject]
) -> Iterator[Tuple[Position, GridObject]]:
    """Utility for iterating *once* over position/objects.

    Every object is only yielded once, even if the objects move during the
    interleaved iteration.
    """

    objects: List[GridObject] = []
    for position in grid.area.positions():
        obj = grid[position]

        if isinstance(obj, object_type) and not any(obj is x for x in objects):
            objects.append(obj)
            yield position, obj


def _step_moving_obstacle(
    grid: Grid, position: Position, *, rng: Optional[rnd.Generator] = None
):
    """Utility for moving a single MovingObstacle randomly"""
    assert isinstance(grid[position], MovingObstacle)

    rng = get_gv_rng_if_none(rng)

    next_positions = [
        next_position
        for next_position in get_manhattan_boundary(position, distance=1)
        if grid.area.contains(next_position)
        and isinstance(grid[next_position], Floor)
    ]

    try:
        next_position = rng.choice(next_positions)
    except ValueError:
        pass
    else:
        grid.swap(position, next_position)


@transition_function_registry.register
def step_moving_obstacles(
    state: State,
    action: Action,
    *,
    rng: Optional[rnd.Generator] = None,
) -> None:
    """Moves moving obstacles randomly

    Moves each MovingObstacle only to cells containing _Floor_ objects, and
    will do so with random walk. In current implementation can only move 1 cell
    non-diagonally. If (and only if) no open cells are available will it stay
    put

    Args:
        state (`State`): current state
        action (`Action`): action taken by agent (ignored)
    """
    rng = get_gv_rng_if_none(rng)

    for position, _ in _unique_object_type_positions(
        state.grid, MovingObstacle
    ):
        _step_moving_obstacle(state.grid, position, rng=rng)


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

    position = state.agent.position_in_front()

    if not state.grid.area.contains(position):
        return

    door = state.grid[position]

    if not isinstance(door, Door):
        return

    if door.is_open:
        pass

    elif not door.locked:
        door.state = Door.Status.OPEN

    else:
        if (
            isinstance(state.agent.obj, Key)
            and state.agent.obj.color == door.color
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

    position = state.agent.position_in_front()

    if not state.grid.area.contains(position):
        return

    box = state.grid[position]

    if not isinstance(box, Box):
        return

    state.grid[position] = box.content


@transition_function_registry.register
def step_telepod(
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
        state.agent.position = rng.choice(positions)


def factory(name: str, **kwargs) -> TransitionFunction:

    if is_custom_function(name):
        name = import_custom_function(name)

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
