from typing import Callable

from gym_gridverse.envs import Actions
from gym_gridverse.grid_object import Goal
from gym_gridverse.state import State

RewardFunction = Callable[[State, Actions, State], float]


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
    state: State,  # pylint: disable=unused-argument
    action: Actions,  # pylint: disable=unused-argument
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
    return (
        reward_on
        if isinstance(next_state.grid[next_state.agent.position], Goal)
        else reward_off
    )
