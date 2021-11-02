from typing import Type

import pytest

from gym_gridverse.action import Action
from gym_gridverse.agent import Agent
from gym_gridverse.envs.terminating_functions import (
    bump_into_wall,
    factory,
    reach_exit,
)
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Exit, Wall
from gym_gridverse.state import State


# TODO: turn into fixture
def make_exit_state(agent_on_exit: bool) -> State:
    """makes a simple state with exit object and agent on or off the exit"""
    grid = Grid.from_shape((2, 1))
    grid[0, 0] = Exit()
    agent_position = Position(0, 0) if agent_on_exit else Position(1, 0)
    agent = Agent(agent_position, Orientation.F)
    return State(grid, agent)


# TODO: turn into fixture
def make_wall_state() -> State:
    """makes a simple state with Wall object and agent in front of it"""
    grid = Grid.from_shape((2, 1))
    grid[0, 0] = Wall()
    agent = Agent(Position(1, 0), Orientation.F)
    return State(grid, agent)


@pytest.mark.parametrize(
    'next_state,expected',
    [
        # on exit
        (make_exit_state(agent_on_exit=True), True),
        # off exit
        (make_exit_state(agent_on_exit=False), False),
    ],
)
def test_reach_exit(next_state: State, expected: bool):
    assert reach_exit(None, None, next_state) == expected


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
    assert bump_into_wall(state, action, None) == expected


# TODO: incorporate this test with previous one
def test_bump_into_wall_special_case():
    state = make_wall_state()
    state.agent.orientation = Orientation.L
    assert bump_into_wall(state, Action.MOVE_RIGHT, None)


@pytest.mark.parametrize(
    'name,kwargs',
    [
        (
            'reduce',
            {
                'terminating_functions': [],
                'reduction': lambda *args, **kwargs: True,
            },
        ),
        ('reduce_any', {'terminating_functions': []}),
        ('reduce_all', {'terminating_functions': []}),
        ('overlap', {'object_type': Exit}),
        ('reach_exit', {}),
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
        ('reduce', {}, ValueError),
        ('reduce_any', {}, ValueError),
        ('reduce_all', {}, ValueError),
        ('overlap', {}, ValueError),
    ],
)
def test_factory_invalid(name: str, kwargs, exception: Type[Exception]):
    with pytest.raises(exception):
        factory(name, **kwargs)
