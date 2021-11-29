import dataclasses

import pytest

from gym_gridverse.agent import Agent
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Color, Floor, Key, NoneGridObject, Wall
from gym_gridverse.state import State
from gym_gridverse.utils.fast_copy import fast_copy


def _change_grid(state: State):
    """changes one object in the grid"""
    state.grid[0, 0] = (
        Wall() if isinstance(state.grid[0, 0], Floor) else Floor()
    )


def _change_agent_position(state: State):
    """changes agent position"""
    state.agent.position = dataclasses.replace(
        state.agent.position,
        y=(state.agent.position.y + 1) % state.grid.shape.height,
        x=(state.agent.position.x + 1) % state.grid.shape.width,
    )


def _change_agent_orientation(state: State):
    """changes agent orientation"""
    state.agent.orientation *= Orientation.B


def _change_agent_grid_object(state: State):
    """changes agent grid object"""
    state.agent.grid_object = (
        Key(Color.RED)
        if isinstance(state.agent.grid_object, NoneGridObject)
        else NoneGridObject()
    )


@pytest.mark.parametrize(
    'state',
    [
        State(
            Grid.from_shape((2, 3)),
            Agent(Position(0, 0), Orientation.F),
        ),
        State(
            Grid.from_shape((3, 2)),
            Agent(Position(1, 1), Orientation.B, Key(Color.RED)),
        ),
    ],
)
def test_state_eq(state: State):
    other_state = fast_copy(state)
    assert state == other_state

    other_state = fast_copy(state)
    _change_grid(other_state)
    assert state != other_state

    other_state = fast_copy(state)
    _change_agent_position(other_state)
    assert state != other_state

    other_state = fast_copy(state)
    _change_agent_orientation(other_state)
    assert state != other_state

    other_state = fast_copy(state)
    _change_agent_grid_object(other_state)
    assert state != other_state


def test_state_hash():
    wall_position = Position(0, 0)
    agent_position = Position(0, 1)
    agent_orientation = Orientation.F
    agent_grid_object = None

    grid = Grid.from_shape((2, 2))
    grid[wall_position] = Wall()
    agent = Agent(agent_position, agent_orientation, agent_grid_object)
    state = State(grid, agent)

    hash(state)
