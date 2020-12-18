import pytest

from gym_gridverse.action import Action
from gym_gridverse.envs.terminating_functions import (
    bump_into_wall,
    factory,
    reach_goal,
)
from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import Goal, Wall
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State


# TODO turn into fixture
def make_goal_state(agent_on_goal: bool) -> State:
    """makes a simple state with goal object and agent on or off the goal"""
    grid = Grid(2, 1)
    grid[0, 0] = Goal()
    agent_position = (0, 0) if agent_on_goal else (1, 0)
    agent = Agent(agent_position, Orientation.N)
    return State(grid, agent)


# TODO turn into fixture
def make_wall_state() -> State:
    """makes a simple state with Wall object and agent in front of it"""
    grid = Grid(2, 1)
    grid[0, 0] = Wall()
    agent = Agent((1, 0), Orientation.N)
    return State(grid, agent)


@pytest.mark.parametrize(
    'next_state,expected',
    [
        # on goal
        (make_goal_state(agent_on_goal=True), True),
        # off goal
        (make_goal_state(agent_on_goal=False), False),
    ],
)
def test_reach_goal(next_state: State, expected: bool):
    assert reach_goal(None, None, next_state) == expected  # type: ignore


@pytest.mark.parametrize(
    'state,action,expected',
    [
        # no bumps
        (make_wall_state(), Action.MOVE_LEFT, False),
        (make_wall_state(), Action.TURN_RIGHT, False),
        (make_wall_state(), Action.ACTUATE, False),
        # bumps
        (make_wall_state(), Action.MOVE_FORWARD, True),
    ],
)
def test_bump_into_wall(state: State, action: Action, expected: bool):
    assert bump_into_wall(state, action, None) == expected  # type: ignore


# TODO incorporate this test with previous one
def test_bump_into_wall_special_case():
    state = make_wall_state()
    state.agent.orientation = Orientation.W
    assert bump_into_wall(state, Action.MOVE_RIGHT, None)


@pytest.mark.parametrize(
    'name,kwargs',
    [
        ('chain_any', {'terminating_functions': []}),
        ('chain_all', {'terminating_functions': []}),
        ('overlap', {'object_type': Goal}),
        ('reach_goal', {}),
        ('bump_moving_obstacle', {}),
        ('bump_into_wall', {}),
    ],
)
def test_factory_valid(name: str, kwargs):
    factory(name, **kwargs)


@pytest.mark.parametrize(
    'name,kwargs,exception',
    [
        ('invalid', {}, ValueError),
        ('chain_any', {}, ValueError),
        ('chain_all', {}, ValueError),
        ('overlap', {}, ValueError),
    ],
)
def test_factory_invalid(name: str, kwargs, exception: Exception):
    with pytest.raises(exception):  # type: ignore
        factory(name, **kwargs)
