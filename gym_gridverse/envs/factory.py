""" Tying the magic together into constructing specific domains """

from functools import partial
from typing import Callable, Dict, List

from gym_gridverse.envs import (
    observation_functions,
    reset_functions,
    reward_functions,
    state_dynamics,
    terminating_functions,
)
from gym_gridverse.envs.env import Environment
from gym_gridverse.envs.minigrid_env import Minigrid


def create_env(
    reset: reset_functions.ResetFunction,
    transition_functions: List[state_dynamics.StateDynamics],
    rewards: List[reward_functions.RewardFunction],
    terminations: List[terminating_functions.TerminatingFunction],
) -> Environment:
    """Factory for environments

    Chains together the transition, reward and termination functions

    * transitions are called in order
    * reward are additive
    * terminates if one of the termination functions return true

    Args:
        reset (`reset_functions.ResetFunction`):
        transition_functions (`List[state_dynamics.StateDynamics]`): called in order
        rewards (`List[reward_functions.RewardFunction]`): Combined additive
        terminations (`List[terminating_functions.TerminatingFunction]`): Called as big 'or'

    Returns:
        Environment: Minigrid environment
    """

    # Transitions are applied in order
    def transition(s, a):
        for f in transition_functions:
            f(s, a)

    # TODO make more general
    observation = observation_functions.minigrid_observation

    # Rewards are additive
    def reward(s, a, next_s):
        return sum(r(s, a, next_s) for r in rewards)

    # Termination is a big or
    def termination(s, a, next_s):
        return any(t(s, a, next_s) for t in terminations)

    return Minigrid(reset, transition, observation, reward, termination)


def plain_navigation_task(
    reset_func: reset_functions.ResetFunction,
) -> Environment:
    """Creates a basic navigation task

    * Empty room
    * 4-room environment

    Args:
        reset_func (reset_functions.ResetFunction):

    Returns:
        Environment: Minigrid with basic navigation dynamics
    """

    transitions: List[state_dynamics.StateDynamics] = [
        state_dynamics.update_agent
    ]
    rewards: List[reward_functions.RewardFunction] = [
        reward_functions.reach_goal
    ]
    terminations: List[terminating_functions.TerminatingFunction] = [
        terminating_functions.reach_goal
    ]

    return create_env(reset_func, transitions, rewards, terminations)


def dynamic_obstacle_minigrid(
    size: int, random_pos: bool, num_obstacles: int
) -> Environment:

    # +2 size to accommodate the walls
    reset_func = partial(
        reset_functions.reset_minigrid_dynamic_obstacles,
        size + 2,
        size + 2,
        num_obstacles,
        random_pos,
    )

    transitions: List[state_dynamics.StateDynamics] = [
        state_dynamics.update_agent,
        state_dynamics.step_objects,
    ]
    rewards: List[reward_functions.RewardFunction] = [
        reward_functions.reach_goal,
        reward_functions.bump_moving_obstacle
        # TODO: crashing into wall?
    ]
    terminations: List[terminating_functions.TerminatingFunction] = [
        terminating_functions.reach_goal,
        terminating_functions.bump_moving_obstacle
        # TODO: crashing into wall?
    ]

    return create_env(reset_func, transitions, rewards, terminations)


def empty_gym_minigrid(size: int, random_pos: bool) -> Environment:
    """Creates the gym-minigrid empty environment of `size` and `random_pos`

    Args:
        size (`int`): The size of x by x floor
        random_pos (`bool`): Whether the agent spawns randomly

    Returns:
        Environment
    """

    # +2 size to accommodate the walls
    reset: reset_functions.ResetFunction = partial(
        reset_functions.reset_minigrid_empty, size + 2, size + 2, random_pos
    )

    return plain_navigation_task(reset)


def four_room() -> Environment:
    """Creates the gym-four-room environment

    Returns:
        Environment: [TODO:description]
    """

    reset: reset_functions.ResetFunction = partial(
        reset_functions.reset_minigrid_four_rooms, 19, 19
    )

    return plain_navigation_task(reset)


STRING_TO_GYM_CONSTRUCTOR: Dict[str, Callable[[], Environment]] = {
    # Empty rooms
    "MiniGrid-Empty-5x5-v0": partial(
        empty_gym_minigrid, size=5, random_pos=False
    ),
    "MiniGrid-Empty-Random-5x5-v0": partial(
        empty_gym_minigrid, size=5, random_pos=True
    ),
    "MiniGrid-Empty-6x6-v0": partial(
        empty_gym_minigrid, size=6, random_pos=False
    ),
    "MiniGrid-Empty-Random-6x6-v0": partial(
        empty_gym_minigrid, size=6, random_pos=True
    ),
    "MiniGrid-Empty-8x8-v0": partial(
        empty_gym_minigrid, size=8, random_pos=False
    ),
    "MiniGrid-Empty-16x16-v0": partial(
        empty_gym_minigrid, size=8, random_pos=False
    ),
    # 4 rooms
    "MiniGrid-FourRooms-v0": partial(four_room),
    # Dynamic obstacle environments
    "MiniGrid-Dynamic-Obstacles-5x5-v0": partial(
        dynamic_obstacle_minigrid, size=5, random_pos=False, num_obstacles=2
    ),
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0": partial(
        dynamic_obstacle_minigrid, size=5, random_pos=True, num_obstacles=2
    ),
    "MiniGrid-Dynamic-Obstacles-6x6-v0": partial(
        dynamic_obstacle_minigrid, size=5, random_pos=False, num_obstacles=3
    ),
    "MiniGrid-Dynamic-Obstacles-Random-6x6-v0": partial(
        dynamic_obstacle_minigrid, size=6, random_pos=True, num_obstacles=3
    ),
    "MiniGrid-Dynamic-Obstacles-8x8-v0": partial(
        dynamic_obstacle_minigrid, size=8, random_pos=False, num_obstacles=4
    ),
    "MiniGrid-Dynamic-Obstacles-16x16-v0": partial(
        dynamic_obstacle_minigrid, size=16, random_pos=False, num_obstacles=8
    ),
}


def gym_minigrid_from_descr(descr: str) -> Environment:
    try:
        return STRING_TO_GYM_CONSTRUCTOR[descr]()
    except KeyError:
        raise ValueError(f"No environment named {descr} is implemented")
