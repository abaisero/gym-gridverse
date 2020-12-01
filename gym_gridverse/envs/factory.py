""" Tying the magic together into constructing specific domains """

from functools import partial
from typing import Callable, Dict, List, Optional, Type

import numpy.random as rnd

from gym_gridverse.actions import Actions
from gym_gridverse.envs import (
    observation_functions,
    reset_functions,
    reward_functions,
    terminating_functions,
    transition_functions,
)
from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.envs.gridworld import GridWorld
from gym_gridverse.geometry import Shape
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Floor,
    Goal,
    GridObject,
    Key,
    MovingObstacle,
    Wall,
)
from gym_gridverse.spaces import (
    ActionSpace,
    DomainSpace,
    ObservationSpace,
    StateSpace,
)
from gym_gridverse.state import State


def create_env(
    domain_space: DomainSpace,
    reset: reset_functions.ResetFunction,
    transition_functions: List[transition_functions.TransitionFunction],
    rewards: List[reward_functions.RewardFunction],
    terminations: List[terminating_functions.TerminatingFunction],
) -> InnerEnv:
    """Factory for environments

    Chains together the transition, reward and termination functions

    * transitions are called in order
    * reward are additive
    * terminates if one of the termination functions return true

    Args:
        domain_space (`DomainSpace`):
        reset (`reset_functions.ResetFunction`):
        transition_functions (`List[transition_functions.TransitionFunction]`): called in order
        rewards (`List[reward_functions.RewardFunction]`): Combined additive
        terminations (`List[terminating_functions.TerminatingFunction]`): Called as big 'or'

    Returns:
        InnerEnv: GridWorld environment
    """

    # Transitions are applied in order
    def transition(
        state: State, action: Actions, *, rng: Optional[rnd.Generator] = None
    ) -> None:
        for f in transition_functions:
            f(state, action, rng=rng)

    # TODO make more general
    observation = partial(
        observation_functions.minigrid_observation,
        observation_space=domain_space.observation_space,
    )

    # Rewards are additive
    def reward(state: State, action: Actions, next_state: State) -> float:
        return sum(r(state, action, next_state) for r in rewards)

    # Termination is a big or
    def termination(state: State, action: Actions, next_state: State) -> bool:
        return any(t(state, action, next_state) for t in terminations)

    return GridWorld(
        domain_space, reset, transition, observation, reward, termination
    )


def plain_navigation_task(
    reset_func: reset_functions.ResetFunction,
) -> InnerEnv:
    """Creates a basic navigation task

    * Empty room
    * 4-room environment

    Args:
        reset_func (reset_functions.ResetFunction):

    Returns:
        InnerEnv: GridWorld with basic navigation dynamics
    """

    transitions: List[transition_functions.TransitionFunction] = [
        transition_functions.update_agent
    ]
    rewards: List[reward_functions.RewardFunction] = [
        reward_functions.reach_goal
    ]
    terminations: List[terminating_functions.TerminatingFunction] = [
        terminating_functions.reach_goal
    ]

    grid_shape = reset_func().grid.shape  # XXX: we hate this
    objects = [Wall, Floor, Goal]
    colors = [Colors.NONE]

    state_space = StateSpace(grid_shape, objects, colors)
    observation_shape = Shape(7, 7)
    observation_space = ObservationSpace(observation_shape, objects, colors)

    # NOTE: here we could limit our actions to original gym interface
    action_space = ActionSpace(list(Actions))

    domain_space = DomainSpace(state_space, action_space, observation_space)

    return create_env(
        domain_space, reset_func, transitions, rewards, terminations
    )


def dynamic_obstacle_minigrid(
    size: int, random_pos: bool, num_obstacles: int
) -> InnerEnv:

    # +2 size to accommodate the walls
    reset_func = partial(
        reset_functions.reset_minigrid_dynamic_obstacles,
        size + 2,
        size + 2,
        num_obstacles,
        random_pos,
    )

    transitions: List[transition_functions.TransitionFunction] = [
        transition_functions.update_agent,
        transition_functions.step_objects,
    ]
    rewards: List[reward_functions.RewardFunction] = [
        reward_functions.reach_goal,
        reward_functions.bump_moving_obstacle,
        reward_functions.bump_into_wall,
    ]
    terminations: List[terminating_functions.TerminatingFunction] = [
        terminating_functions.reach_goal,
        terminating_functions.bump_moving_obstacle,
        terminating_functions.bump_into_wall,
    ]

    grid_shape = reset_func().grid.shape  # XXX: we hate this
    objects = [Wall, Floor, Goal, MovingObstacle]
    colors = [Colors.NONE]

    state_space = StateSpace(grid_shape, objects, colors)
    # TODO: hard-coded observation shape
    observation_space = ObservationSpace(Shape(7, 7), objects, colors)

    # NOTE: here we could limit our actions to original gym interface
    action_space = ActionSpace(list(Actions))

    domain_space = DomainSpace(state_space, action_space, observation_space)
    return create_env(
        domain_space, reset_func, transitions, rewards, terminations
    )


def gym_minigrid_empty(size: int, random_pos: bool) -> InnerEnv:
    """Creates the gym-minigrid empty environment of `size` and `random_pos`

    Args:
        size (`int`): The size of x by x floor
        random_pos (`bool`): Whether the agent spawns randomly

    Returns:
        InnerEnv
    """

    # +2 size to accommodate the walls
    reset = partial(
        reset_functions.reset_minigrid_empty, size + 2, size + 2, random_pos
    )

    return plain_navigation_task(reset)


def gym_minigrid_four_room() -> InnerEnv:
    """Creates the gym-four-room environment

    Returns:
        InnerEnv:
    """

    reset = partial(reset_functions.reset_minigrid_rooms, 19, 19, layout=(2, 2))

    return plain_navigation_task(reset)


def gym_door_key_env(size: int) -> InnerEnv:
    """Creates the 'door key' gym environment

    Args:
        size (`int`): size of the (rectangular) grid

    Returns:
        InnerEnv:
    """

    reset = partial(reset_functions.reset_minigrid_door_key, grid_size=size + 2)

    transitions: List[transition_functions.TransitionFunction] = [
        transition_functions.update_agent,
        transition_functions.actuate_mechanics,
        transition_functions.pickup_mechanics,
    ]
    rewards: List[reward_functions.RewardFunction] = [
        reward_functions.reach_goal
    ]
    terminations: List[terminating_functions.TerminatingFunction] = [
        terminating_functions.reach_goal
    ]

    grid_shape = reset().grid.shape  # XXX: we hate this
    objects: List[Type[GridObject]] = [Wall, Floor, Goal, Door, Key]
    colors = [Colors.NONE, Colors.YELLOW]
    observation_shape = Shape(7, 7)

    state_space = StateSpace(grid_shape, objects, colors)
    observation_space = ObservationSpace(observation_shape, objects, colors)
    action_space = ActionSpace(list(Actions))

    domain_space = DomainSpace(state_space, action_space, observation_space)

    return create_env(domain_space, reset, transitions, rewards, terminations)


STRING_TO_GYM_CONSTRUCTOR: Dict[str, Callable[[], InnerEnv]] = {
    # Empty rooms
    "MiniGrid-Empty-5x5-v0": partial(
        gym_minigrid_empty, size=5, random_pos=False
    ),
    "MiniGrid-Empty-Random-5x5-v0": partial(
        gym_minigrid_empty, size=5, random_pos=True
    ),
    "MiniGrid-Empty-6x6-v0": partial(
        gym_minigrid_empty, size=6, random_pos=False
    ),
    "MiniGrid-Empty-Random-6x6-v0": partial(
        gym_minigrid_empty, size=6, random_pos=True
    ),
    "MiniGrid-Empty-8x8-v0": partial(
        gym_minigrid_empty, size=8, random_pos=False
    ),
    "MiniGrid-Empty-16x16-v0": partial(
        gym_minigrid_empty, size=16, random_pos=False
    ),
    # 4 rooms
    "MiniGrid-FourRooms-v0": partial(gym_minigrid_four_room),
    # Dynamic obstacle environments
    "MiniGrid-Dynamic-Obstacles-5x5-v0": partial(
        dynamic_obstacle_minigrid, size=5, random_pos=False, num_obstacles=2
    ),
    "MiniGrid-Dynamic-Obstacles-Random-5x5-v0": partial(
        dynamic_obstacle_minigrid, size=5, random_pos=True, num_obstacles=2
    ),
    "MiniGrid-Dynamic-Obstacles-6x6-v0": partial(
        dynamic_obstacle_minigrid, size=6, random_pos=False, num_obstacles=3
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
    "MiniGrid-DoorKey-5x5-v0": partial(gym_door_key_env, size=5),
    "MiniGrid-DoorKey-6x6-v0": partial(gym_door_key_env, size=6),
    "MiniGrid-DoorKey-8x8-v0": partial(gym_door_key_env, size=8),
    "MiniGrid-DoorKey-16x16-v0": partial(gym_door_key_env, size=16),
}


def gym_minigrid_from_descr(descr: str) -> InnerEnv:
    try:
        return STRING_TO_GYM_CONSTRUCTOR[descr]()
    except KeyError:
        raise ValueError(f"No environment named {descr} is implemented")
