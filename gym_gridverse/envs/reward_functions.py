from functools import partial
from typing import Callable, Optional, Sequence, Type

import more_itertools as mitt

from gym_gridverse.actions import Actions
from gym_gridverse.envs.utils import updated_agent_position_if_unobstructed
from gym_gridverse.geometry import DistanceFunction, Position
from gym_gridverse.grid_object import Goal, GridObject, MovingObstacle, Wall
from gym_gridverse.state import State

RewardFunction = Callable[[State, Actions, State], float]
"""Signature that all reward functions must follow"""


def chain(
    state: State,
    action: Actions,
    next_state: State,
    *,
    reward_functions: Sequence[RewardFunction],
) -> float:
    """utility reward function which sums other reward functions

    Args:
        state (`State`):
        action (`Actions`):
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
    action: Actions,  # pylint: disable=unused-argument
    next_state: State,
    *,
    object_type: Type[GridObject],
    reward_on: float = 1.0,
    reward_off: float = 0.0,
) -> float:
    """reward for the agent occupying the same position as another object

    Args:
        state (`State`):
        action (`Actions`):
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
    action: Actions,  # pylint: disable=unused-argument
    next_state: State,  # pylint: disable=unused-argument
    *,
    reward: float = -1.0,
) -> float:
    """a living reward which does not depend on states or actions

    Args:
        state (`State`):
        action (`Actions`):
        next_state (`State`):
        reward (`float`): reward for when agent is on goal

    Returns:
        float: the input reward
    """
    return reward


def reach_goal(
    state: State,
    action: Actions,
    next_state: State,
    *,
    reward_on: float = 1.0,
    reward_off: float = 0.0,
) -> float:
    """reward for the Agent being on a Goal

    Args:
        state (`State`):
        action (`Actions`):
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
    state: State, action: Actions, next_state: State, *, reward: float = -1.0
) -> float:
    """reward for the Agent bumping into on a MovingObstacle

    Args:
        state (`State`):
        action (`Actions`):
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
    action: Actions,  # pylint: disable=unused-argument
    next_state: State,
    *,
    distance_function: DistanceFunction = Position.manhattan_distance,
    object_type: Type[GridObject],
    reward_per_unit_distance: float = -1.0,
) -> float:
    """reward proportional to distance to object

    Args:
        state (`State`):
        action (`Actions`):
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
    action: Actions,  # pylint: disable=unused-argument
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
        action (`Actions`):
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
    action: Actions,
    next_state: State,  # pylint: disable=unused-argument
    *,
    reward: float = -1.0,
):
    """Returns `reward` when bumping into wall, otherwise 0

    Bumping is tested by seeing whether the intended move would end up with the
    agent on a wall.

    Args:
        state (State):
        action (Actions):
        next_state (State):
        reward (float, optional): The reward to provide if bumping into wall
    """

    attempted_next_position = updated_agent_position_if_unobstructed(
        state.agent.position, state.agent.orientation, action
    )

    if isinstance(state.grid[attempted_next_position], Wall):
        return reward

    return 0.0


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

    raise ValueError('invalid reward function name {name}')
