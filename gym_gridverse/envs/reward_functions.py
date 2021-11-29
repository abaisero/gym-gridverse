import inspect
import warnings
from collections import deque
from functools import lru_cache, partial
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Type

import more_itertools as mitt
import numpy as np
import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.action import Action
from gym_gridverse.envs.utils import get_next_position
from gym_gridverse.geometry import DistanceFunction, Position
from gym_gridverse.grid_object import (
    Beacon,
    Door,
    Exit,
    GridObject,
    MovingObstacle,
    Wall,
)
from gym_gridverse.state import State
from gym_gridverse.utils.custom import import_if_custom
from gym_gridverse.utils.functions import checkraise_kwargs, select_kwargs
from gym_gridverse.utils.protocols import (
    get_keyword_parameter,
    get_positional_parameters,
)
from gym_gridverse.utils.registry import FunctionRegistry


class RewardFunction(Protocol):
    """Signature that all reward functions must follow"""

    def __call__(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> float:
        ...


RewardReductionFunction = Callable[[Iterator[float]], float]
"""Signature for a float reduction function"""


class RewardFunctionRegistry(FunctionRegistry):
    def get_protocol_parameters(
        self, signature: inspect.Signature
    ) -> List[inspect.Parameter]:
        state, action, next_state = get_positional_parameters(signature, 3)
        rng = get_keyword_parameter(signature, 'rng')
        return [state, action, next_state, rng]

    def check_signature(self, function: RewardFunction):
        signature = inspect.signature(function)
        state, action, next_state, rng = self.get_protocol_parameters(signature)

        # checks first 3 arguments are positional
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

        if next_state.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise TypeError(
                f'The third argument ({next_state.name}) '
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

        if next_state.annotation not in [inspect.Parameter.empty, State]:
            warnings.warn(
                f'The third argument ({next_state.name}) '
                f'of a registered reward function ({function}) '
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

        if signature.return_annotation not in [inspect.Parameter.empty, float]:
            warnings.warn(
                f'The return type of a registered reward function ({function}) '
                f'has an annotation ({signature.return_annotation}) '
                'which is not `float`.'
            )


reward_function_registry = RewardFunctionRegistry()
"""Reward function registry"""


@reward_function_registry.register
def reduce(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_functions: Sequence[RewardFunction],
    reduction: RewardReductionFunction,
    rng: Optional[rnd.Generator] = None,
) -> float:
    """reduction of multiple reward functions into a single boolean value

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        reward_functions (`Sequence[RewardFunction]`):
        reduction (`RewardReductionFunction`):
        rng (`Generator, optional`)

    Returns:
        bool: reduction operator over the input reward functions
    """
    # TODO: test

    return reduction(
        reward_function(state, action, next_state, rng=rng)
        for reward_function in reward_functions
    )


@reward_function_registry.register
def reduce_sum(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_functions: Sequence[RewardFunction],
    rng: Optional[rnd.Generator] = None,
) -> float:
    """utility reward function which sums other reward functions

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        reward_functions (`Sequence[RewardFunction]`):
        rng (`Generator, optional`)

    Returns:
        float: sum of the evaluated input reward functions
    """
    # TODO: test
    return reduce(
        state,
        action,
        next_state,
        reward_functions=reward_functions,
        reduction=sum,
        rng=rng,
    )


@reward_function_registry.register
def overlap(
    state: State,
    action: Action,
    next_state: State,
    *,
    object_type: Type[GridObject],
    reward_on: float = 1.0,
    reward_off: float = 0.0,
    rng: Optional[rnd.Generator] = None,
) -> float:
    """reward for the agent occupying the same position as another object

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        object_type (`Type[GridObject]`):
        reward_on (`float`): reward for when agent is on the object
        reward_off (`float`): reward for when agent is not on the object
        rng (`Generator, optional`)

    Returns:
        float: one of the two input rewards
    """
    return (
        reward_on
        if isinstance(next_state.grid[next_state.agent.position], object_type)
        else reward_off
    )


@reward_function_registry.register
def living_reward(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward: float = -1.0,
    rng: Optional[rnd.Generator] = None,
) -> float:
    """a living reward which does not depend on states or actions

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        reward (`float`): reward for when agent is on exit
        rng (`Generator, optional`)

    Returns:
        float: the input reward
    """
    return reward


@reward_function_registry.register
def reach_exit(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_on: float = 1.0,
    reward_off: float = 0.0,
    rng: Optional[rnd.Generator] = None,
) -> float:
    """reward for the Agent being on a Exit

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        reward_on (`float`): reward for when agent is on exit
        reward_off (`float`): reward for when agent is not on exit
        rng (`Generator, optional`)

    Returns:
        float: one of the two input rewards
    """
    return overlap(
        state,
        action,
        next_state,
        object_type=Exit,
        reward_on=reward_on,
        reward_off=reward_off,
        rng=rng,
    )


@reward_function_registry.register
def bump_moving_obstacle(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward: float = -1.0,
    rng: Optional[rnd.Generator] = None,
) -> float:
    """reward for the Agent bumping into on a MovingObstacle

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        reward (`float`): reward for when Agent bumps a MovingObstacle
        rng (`Generator, optional`)

    Returns:
        float: the input reward or 0.0
    """
    return overlap(
        state,
        action,
        next_state,
        object_type=MovingObstacle,
        reward_on=reward,
        reward_off=0.0,
        rng=rng,
    )


@reward_function_registry.register
def proportional_to_distance(
    state: State,
    action: Action,
    next_state: State,
    *,
    distance_function: DistanceFunction = Position.manhattan_distance,
    object_type: Type[GridObject],
    reward_per_unit_distance: float = -1.0,
    rng: Optional[rnd.Generator] = None,
) -> float:
    """reward proportional to distance to object

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        distance_function (`DistanceFunction`):
        object_type: (`Type[GridObject]`): type of unique object in grid
        reward (`float`): reward per unit distance
        rng (`Generator, optional`)

    Returns:
        float: input reward times distance to object
    """

    object_position = mitt.one(
        position
        for position in next_state.grid.area.positions()
        if isinstance(next_state.grid[position], object_type)
    )
    distance = distance_function(next_state.agent.position, object_position)
    return reward_per_unit_distance * distance


@reward_function_registry.register
def getting_closer(
    state: State,
    action: Action,
    next_state: State,
    *,
    distance_function: DistanceFunction = Position.manhattan_distance,
    object_type: Type[GridObject],
    reward_closer: float = 1.0,
    reward_further: float = -1.0,
    rng: Optional[rnd.Generator] = None,
) -> float:
    """reward for getting closer or further to object

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        distance_function (`DistanceFunction`):
        object_type: (`Type[GridObject]`): type of unique object in grid
        reward_closer (`float`): reward for when agent gets closer to object
        reward_further (`float`): reward for when agent gets further to object
        rng (`Generator, optional`)

    Returns:
        float: one of the input rewards, or 0.0 if distance has not changed
    """

    def _distance_agent_object(state):
        object_position = mitt.one(
            position
            for position in state.grid.area.positions()
            if isinstance(state.grid[position], object_type)
        )
        return distance_function(state.agent.position, object_position)

    distance_prev = _distance_agent_object(state)
    distance_next = _distance_agent_object(next_state)

    return (
        reward_closer
        if distance_next < distance_prev
        else reward_further
        if distance_next > distance_prev
        else 0.0
    )


@lru_cache(maxsize=10)
def dijkstra(
    layout: Tuple[Tuple[bool]], source_position: Tuple[int, int]
) -> np.ndarray:
    layout_array = np.array(layout)

    visited = np.zeros(layout_array.shape, dtype=bool)
    visited[source_position] = True
    distances = np.full(layout_array.shape, float('inf'))
    distances[source_position] = 0.0

    frontier = deque([source_position])
    while frontier:
        y_old, x_old = frontier.popleft()

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            y_new = y_old + dy
            x_new = x_old + dx

            if (
                0 <= y_new < layout_array.shape[0]
                and 0 <= x_new < layout_array.shape[1]
                and layout_array[y_new, x_new]
                and not visited[y_new, x_new]
            ):
                distances[y_new, x_new] = distances[y_old, x_old] + 1
                visited[y_new, x_new] = True
                frontier.append((y_new, x_new))

    return distances


@reward_function_registry.register
def getting_closer_shortest_path(
    state: State,
    action: Action,
    next_state: State,
    *,
    object_type: Type[GridObject],
    reward_closer: float = 1.0,
    reward_further: float = -1.0,
    rng: Optional[rnd.Generator] = None,
) -> float:
    """reward for getting closer or further to object, *assuming normal navigation dynamics*

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        object_type: (`Type[GridObject]`): type of unique object in grid
        reward_closer (`float`): reward for when agent gets closer to object
        reward_further (`float`): reward for when agent gets further to object
        rng (`Generator, optional`)

    Returns:
        float: one of the input rewards, or 0.0 if distance has not changed
    """

    def _distance_agent_object(state):
        object_position = mitt.one(
            position
            for position in state.grid.area.positions()
            if isinstance(state.grid[position], object_type)
        )

        layout = tuple(
            tuple(
                not state.grid[y, x].blocks_movement
                for x in range(state.grid.shape.width)
            )
            for y in range(state.grid.shape.height)
        )
        distance_array = dijkstra(
            layout, (object_position.y, object_position.x)
        )
        return distance_array[state.agent.position.y, state.agent.position.x]

    distance_prev = _distance_agent_object(state)
    distance_next = _distance_agent_object(next_state)

    return (
        reward_closer
        if distance_next < distance_prev
        else reward_further
        if distance_next > distance_prev
        else 0.0
    )


@reward_function_registry.register
def bump_into_wall(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward: float = -1.0,
    rng: Optional[rnd.Generator] = None,
):
    """Returns `reward` when bumping into wall, otherwise 0

    Bumping is tested by seeing whether the intended move would end up with the
    agent on a wall.

    Args:
        state (State):
        action (Action):
        next_state (State):
        reward (float): (optional) The reward to provide if bumping into wall
        rng (`Generator, optional`)
    """

    next_position = get_next_position(
        state.agent.position, state.agent.orientation, action
    )

    return (
        reward
        if state.grid.area.contains(next_position)
        and isinstance(state.grid[next_position], Wall)
        else 0.0
    )


@reward_function_registry.register
def actuate_door(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_open: float = 1.0,
    reward_close: float = -1.0,
    rng: Optional[rnd.Generator] = None,
):
    """Returns `reward_open` when opening and `reward_close` when closing door.

    Opening/closing is checked by making sure the actuate action is performed,
    and checking the status of the door in front of the agent.

    Args:
        state (State):
        action (Action):
        next_state (State):
        reward_open (float): (optional) The reward to provide if opening a door
        reward_close (float): (optional) The reward to provide if closing a door
        rng (`Generator, optional`)
    """

    if action is not Action.ACTUATE:
        return 0.0

    position = state.agent.front()

    door = state.grid[position]
    if not isinstance(door, Door):
        return 0.0

    # assumes same door
    next_door = next_state.grid[position]
    if not isinstance(next_door, Door):
        return 0.0

    return (
        reward_open
        if not door.is_open and next_door.is_open
        else reward_close
        if door.is_open and not next_door.is_open
        else 0.0
    )


@reward_function_registry.register
def pickndrop(
    state: State,
    action: Action,
    next_state: State,
    *,
    object_type: Type[GridObject],
    reward_pick: float = 1.0,
    reward_drop: float = -1.0,
    rng: Optional[rnd.Generator] = None,
):
    """Returns `reward_pick` / `reward_drop` when an object is picked / dropped.

    Picking/dropping is checked by the agent's object, and not the action.

    Args:
        state (State):
        action (Action):
        next_state (State):
        reward_pick (float): (optional) The reward to provide if picking a key
        reward_drop (float): (optional) The reward to provide if dropping a key
        rng (`Generator, optional`)
    """

    has_key = isinstance(state.agent.grid_object, object_type)
    next_has_key = isinstance(next_state.agent.grid_object, object_type)

    return (
        reward_pick
        if not has_key and next_has_key
        else reward_drop
        if has_key and not next_has_key
        else 0.0
    )


@reward_function_registry.register
def reach_exit_memory(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_good: float = 1.0,
    reward_bad: float = -1.0,
    rng: Optional[rnd.Generator] = None,
) -> float:
    """reward for the Agent being on a Exit

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        reward_good (`float`): reward for when agent is on the good exit
        reward_bad (`float`): reward for when agent is on the bad exit
        rng (`Generator, optional`)

    Returns:
        float: one of the two input rewards
    """
    # TODO: test

    agent_grid_object = next_state.grid[next_state.agent.position]
    grid_objects = (
        next_state.grid[position]
        for position in next_state.grid.area.positions()
    )
    beacon_color = next(
        grid_object.color
        for grid_object in grid_objects
        if isinstance(grid_object, Beacon)
    )

    return (
        (reward_good if agent_grid_object.color is beacon_color else reward_bad)
        if isinstance(agent_grid_object, Exit)
        else 0.0
    )


def factory(name: str, **kwargs) -> RewardFunction:
    name = import_if_custom(name)

    try:
        function = reward_function_registry[name]
    except KeyError as error:
        raise ValueError(f'invalid reward function name {name}') from error

    signature = inspect.signature(function)
    required_keys = [
        parameter.name
        for parameter in reward_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is inspect.Parameter.empty
    ]
    optional_keys = [
        parameter.name
        for parameter in reward_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is not inspect.Parameter.empty
    ]

    checkraise_kwargs(kwargs, required_keys)
    kwargs = select_kwargs(kwargs, required_keys + optional_keys)
    return partial(function, **kwargs)
