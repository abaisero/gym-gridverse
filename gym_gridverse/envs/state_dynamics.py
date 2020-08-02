""" Functions to model dynamics """

from typing import Callable

from gym_gridverse.envs import Actions
from gym_gridverse.geometry import Position
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State

TRANSLATION_ACTIONS = [
    Actions.MOVE_FORWARD,
    Actions.MOVE_BACKWARD,
    Actions.MOVE_LEFT,
    Actions.MOVE_RIGHT,
]

ROTATION_ACTIONS = [
    Actions.TURN_LEFT,
    Actions.TURN_RIGHT,
]


StateDynamics = Callable[[State, Actions], None]


def move_agent(agent: Agent, grid: Grid, action: Actions) -> None:
    """Applies translation to agent (e.g. up/down/left/right)

    Leaves the state unaffected if any other action was taken instead

    Args:
        agent (`Agent`):
        grid (`Grid`):
        action (`Actions`):

    Returns:
        None:
    """

    if action not in TRANSLATION_ACTIONS:
        return

    # Map directions to relative orientation
    direction_to_relative_orientation = {
        Actions.MOVE_FORWARD: agent.orientation,
        Actions.MOVE_LEFT: agent.orientation.rotate_left(),
        Actions.MOVE_RIGHT: agent.orientation.rotate_right(),
        Actions.MOVE_BACKWARD: agent.orientation.rotate_right().rotate_right(),
    }

    delta = direction_to_relative_orientation[action].as_delta_position()
    next_pos = Position.add(agent.position, delta)

    # exit if next position is not legal
    if next_pos not in grid or grid[next_pos].blocks:
        return

    agent.position = next_pos


def rotate_agent(agent: Agent, action: Actions) -> None:
    """Rotates agent according to action (e.g. turn left/right)

    Leaves the state unaffected if any other action was taken instead

    TODO: test

    Args:
        agent (`Agent`):
        action (`Actions`):

    Returns:
        None:
    """

    if action not in ROTATION_ACTIONS:
        return

    if action == Actions.TURN_LEFT:
        agent.orientation = agent.orientation.rotate_left()

    if action == Actions.TURN_RIGHT:
        agent.orientation = agent.orientation.rotate_right()


def update_agent(state: State, action: Actions) -> None:
    """Simply updates the agents location and orientation based on action

    If action does not affect this (e.g. not turning not moving), then leaves
    the state untouched

    Args:
        state (`State`):
        action (`Actions`):

    Returns:
        None
    """
    if action not in TRANSLATION_ACTIONS and action not in ROTATION_ACTIONS:
        return

    if action in ROTATION_ACTIONS:
        rotate_agent(state.agent, action)

    if action in TRANSLATION_ACTIONS:
        translate_agent(state.agent, state.grid, action)


def step_objects(state: State, action: Actions) -> None:
    """Calls `step` on all the objects in the grid

    Args:
        state (`State`):
        action (`Actions`):

    Returns:
        None:
    """
    for pos in state.grid.positions():
        state.grid[pos].step(state, action)
