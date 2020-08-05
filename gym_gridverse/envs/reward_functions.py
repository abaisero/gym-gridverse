from typing import Callable, Sequence, Type

import more_itertools as mitt

from gym_gridverse.actions import Actions
from gym_gridverse.geometry import DistanceFunction, Position
from gym_gridverse.grid_object import Goal, GridObject, MovingObstacle
from gym_gridverse.state import State

RewardFunction = Callable[[State, Actions, State], float]


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
    state: State, action: Actions, next_state: State, *, reward: float = -1.0,
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
