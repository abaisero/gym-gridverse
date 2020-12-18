from functools import partial
from typing import Callable, Optional, Sequence, Type

import more_itertools as mitt

from gym_gridverse.action import Action
from gym_gridverse.envs.utils import updated_agent_position_if_unobstructed
from gym_gridverse.geometry import DistanceFunction, Position
from gym_gridverse.grid_object import (
    Door,
    Goal,
    GridObject,
    MovingObstacle,
    Wall,
)
from gym_gridverse.state import State

RewardFunction = Callable[[State, Action, State], float]
"""Signature that all reward functions must follow"""


def chain(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_functions: Sequence[RewardFunction],
) -> float:
    """utility reward function which sums other reward functions

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        reward_functions (`Sequence[RewardFunction]`):

    Returns:
        float: sum of the evaluated input reward functions
    """
    return sum(
        reward_function(state, action, next_state)
        for reward_function in reward_functions
    )


def overlap(
    state: State,  # pylint: disable=unused-argument
    action: Action,  # pylint: disable=unused-argument
    next_state: State,
    *,
    object_type: Type[GridObject],
    reward_on: float = 1.0,
    reward_off: float = 0.0,
) -> float:
    """reward for the agent occupying the same position as another object

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        object_type (`Type[GridObject]`):
        reward_on (`float`): reward for when agent is on the object
        reward_off (`float`): reward for when agent is not on the object

    Returns:
        float: one of the two input rewards
    """
    return (
        reward_on
        if isinstance(next_state.grid[next_state.agent.position], object_type)
        else reward_off
    )


def living_reward(
    state: State,  # pylint: disable=unused-argument
    action: Action,  # pylint: disable=unused-argument
    next_state: State,  # pylint: disable=unused-argument
    *,
    reward: float = -1.0,
) -> float:
    """a living reward which does not depend on states or actions

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        reward (`float`): reward for when agent is on goal

    Returns:
        float: the input reward
    """
    return reward


def reach_goal(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_on: float = 1.0,
    reward_off: float = 0.0,
) -> float:
    """reward for the Agent being on a Goal

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        reward_on (`float`): reward for when agent is on goal
        reward_off (`float`): reward for when agent is not on goal

    Returns:
        float: one of the two input rewards
    """
    return overlap(
        state,
        action,
        next_state,
        object_type=Goal,
        reward_on=reward_on,
        reward_off=reward_off,
    )


def bump_moving_obstacle(
    state: State, action: Action, next_state: State, *, reward: float = -1.0
) -> float:
    """reward for the Agent bumping into on a MovingObstacle

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        reward (`float`): reward for when Agent bumps a MovingObstacle

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
    )


def proportional_to_distance(
    state: State,  # pylint: disable=unused-argument
    action: Action,  # pylint: disable=unused-argument
    next_state: State,
    *,
    distance_function: DistanceFunction = Position.manhattan_distance,
    object_type: Type[GridObject],
    reward_per_unit_distance: float = -1.0,
) -> float:
    """reward proportional to distance to object

    Args:
        state (`State`):
        action (`Action`):
        next_state (`State`):
        distance_function (`DistanceFunction`):
        object_type: (`Type[GridObject]`): type of unique object in grid
        reward (`float`): reward per unit distance

    Returns:
        float: input reward times distance to object
    """

    object_position = mitt.one(
        position
        for position in next_state.grid.positions()
        if isinstance(next_state.grid[position], object_type)
    )
    distance = distance_function(next_state.agent.position, object_position)
    return reward_per_unit_distance * distance


def getting_closer(
    state: State,
    action: Action,  # pylint: disable=unused-argument
    next_state: State,
    *,
    distance_function: DistanceFunction = Position.manhattan_distance,
    object_type: Type[GridObject],
    reward_closer: float = 1.0,
    reward_further: float = -1.0,
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

    Returns:
        float: one of the input rewards, or 0.0 if distance has not changed
    """

    def _distance_agent_object(state):
        object_position = mitt.one(
            position
            for position in state.grid.positions()
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


def bump_into_wall(
    state: State,
    action: Action,
    next_state: State,  # pylint: disable=unused-argument
    *,
    reward: float = -1.0,
):
    """Returns `reward` when bumping into wall, otherwise 0

    Bumping is tested by seeing whether the intended move would end up with the
    agent on a wall.

    Args:
        state (State):
        action (Action):
        next_state (State):
        reward (float): (optional) The reward to provide if bumping into wall
    """

    attempted_next_position = updated_agent_position_if_unobstructed(
        state.agent.position, state.agent.orientation, action
    )

    return (
        reward
        if attempted_next_position in state.grid
        and isinstance(state.grid[attempted_next_position], Wall)
        else 0.0
    )


def actuate_door(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_open: float = 1.0,
    reward_close: float = -1.0,
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
    """

    if action is not Action.ACTUATE:
        return 0.0

    position = state.agent.position_in_front()

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


def pickndrop(
    state: State,
    action: Action,  # pylint: disable=unused-argument
    next_state: State,
    *,
    object_type: Type[GridObject],
    reward_pick: float = 1.0,
    reward_drop: float = -1.0,
):
    """Returns `reward_pick` / `reward_drop` when an object is picked / dropped.

    Picking/dropping is checked by the agent's object, and not the action.

    Args:
        state (State):
        action (Action):
        next_state (State):
        reward_pick (float): (optional) The reward to provide if picking a key
        reward_drop (float): (optional) The reward to provide if dropping a key
    """

    has_key = isinstance(state.agent.obj, object_type)
    next_has_key = isinstance(next_state.agent.obj, object_type)

    return (
        reward_pick
        if not has_key and next_has_key
        else reward_drop
        if has_key and not next_has_key
        else 0.0
    )


def factory(  # pylint: disable=too-many-branches
    name: str,
    *,
    reward_functions: Optional[Sequence[RewardFunction]] = None,
    reward: Optional[float] = None,
    reward_on: Optional[float] = None,
    reward_off: Optional[float] = None,
    object_type: Optional[Type[GridObject]] = None,
    distance_function: Optional[DistanceFunction] = None,
    reward_per_unit_distance: Optional[float] = None,
    reward_closer: Optional[float] = None,
    reward_further: Optional[float] = None,
    reward_open: Optional[float] = None,
    reward_close: Optional[float] = None,
    reward_pick: Optional[float] = None,
    reward_drop: Optional[float] = None,
) -> RewardFunction:

    if name == 'chain':
        if None in [reward_functions]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(chain, reward_functions=reward_functions)

    if name == 'overlap':
        if None in [object_type, reward_on, reward_off]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            overlap,
            object_type=object_type,
            reward_on=reward_on,
            reward_off=reward_off,
        )

    if name == 'living_reward':
        if None in [reward]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(living_reward, reward=reward)

    if name == 'reach_goal':
        if None in [reward_on, reward_off]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(reach_goal, reward_on=reward_on, reward_off=reward_off)

    if name == 'bump_moving_obstacle':
        if None in [reward]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(bump_moving_obstacle, reward=reward)

    if name == 'proportional_to_distance':
        if None in [distance_function, object_type, reward_per_unit_distance]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            proportional_to_distance,
            distance_function=distance_function,
            object_type=object_type,
            reward_per_unit_distance=reward_per_unit_distance,
        )

    if name == 'getting_closer':
        if None in [
            distance_function,
            object_type,
            reward_closer,
            reward_further,
        ]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            getting_closer,
            distance_function=distance_function,
            object_type=object_type,
            reward_closer=reward_closer,
            reward_further=reward_further,
        )

    if name == 'bump_into_wall':
        if None in [reward]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(bump_into_wall, reward=reward)

    if name == 'actuate_door':
        if None in [reward_open, reward_close]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            actuate_door, reward_open=reward_open, reward_close=reward_close
        )

    if name == 'pickndrop':
        if None in [object_type, reward_pick, reward_drop]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            pickndrop,
            object_type=object_type,
            reward_pick=reward_pick,
            reward_drop=reward_drop,
        )

    raise ValueError('invalid reward function name {name}')
