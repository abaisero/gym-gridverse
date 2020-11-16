""" Functions to model dynamics """

from typing import Callable, List

from gym_gridverse.actions import ROTATION_ACTIONS, TRANSLATION_ACTIONS, Actions
from gym_gridverse.envs.utils import updated_agent_position_if_unobstructed
from gym_gridverse.grid_object import Floor, GridObject, NoneGridObject
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State

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

    next_pos = updated_agent_position_if_unobstructed(
        agent.position, agent.orientation, action
    )

    # exit if next position is not legal
    if next_pos not in grid or grid[next_pos].blocks:
        return

    agent.position = next_pos


def rotate_agent(agent: Agent, action: Actions) -> None:
    """Rotates agent according to action (e.g. turn left/right)

    Leaves the state unaffected if any other action was taken instead

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
        move_agent(state.agent, state.grid, action)


def step_objects(state: State, action: Actions) -> None:
    """Calls `step` on all the objects in the grid

    Args:
        state (`State`):
        action (`Actions`):

    Returns:
        None:
    """

    stepped_objects: List[GridObject] = []

    for pos in state.grid.positions():
        obj = state.grid[pos]

        if not any([x is obj for x in stepped_objects]):
            stepped_objects.append(obj)
            obj.step(state, action)


def actuate_mechanics(state: State, action: Actions) -> None:
    """Implements the mechanics of actuation

    Calls obj.actuate(state) on the object in front of agent

    Args:
        state (`State`):
        action (`Actions`):

    Returns:
        None:
    """

    if action == Actions.ACTUATE:
        obj_in_front_of_agent = state.grid[state.agent.position_in_front()]
        obj_in_front_of_agent.actuate(state)


def pickup_mechanics(state: State, action: Actions) -> None:
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
        action (`Actions`):

    Returns:
        None:
    """

    if action != Actions.PICK_N_DROP:
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


def factory(name: str) -> StateDynamics:

    if name == 'update_agent':
        return update_agent

    if name == 'step_objects':
        return step_objects

    if name == 'actuate_mechanics':
        return actuate_mechanics

    if name == 'pickup_mechanics':
        return pickup_mechanics

    raise ValueError(f'invalid transition function name `{name}`')
