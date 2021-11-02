import dataclasses
from copy import deepcopy

import pytest

from gym_gridverse.agent import Agent
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Color, Floor, Key, NoneGridObject, Wall
from gym_gridverse.observation import Observation


def _change_grid(observation: Observation):
    """changes one object in the grid"""
    observation.grid[Position(0, 0)] = (
        Wall()
        if isinstance(observation.grid[Position(0, 0)], Floor)
        else Floor()
    )


def _change_agent_position(observation: Observation):
    """changes agent position"""
    observation.agent.position = dataclasses.replace(
        observation.agent.position,
        y=(observation.agent.position.y + 1) % observation.grid.shape.height,
        x=(observation.agent.position.x + 1) % observation.grid.shape.width,
    )


def _change_agent_orientation(observation: Observation):
    """changes agent orientation"""
    observation.agent.orientation *= Orientation.BACKWARD


def _change_agent_object(observation: Observation):
    """changes agent object"""
    observation.agent.obj = (
        Key(Color.RED)
        if isinstance(observation.agent.obj, NoneGridObject)
        else NoneGridObject()
    )


@pytest.mark.parametrize(
    'observation',
    [
        Observation(
            Grid.from_shape((2, 3)),
            Agent(Position(0, 0), Orientation.FORWARD),
        ),
        Observation(
            Grid.from_shape((3, 2)),
            Agent(Position(1, 1), Orientation.BACKWARD, Key(Color.RED)),
        ),
    ],
)
def test_observation_eq(observation: Observation):
    other_observation = deepcopy(observation)
    assert observation == other_observation

    other_observation = deepcopy(observation)
    _change_grid(other_observation)
    assert observation != other_observation

    other_observation = deepcopy(observation)
    _change_agent_position(other_observation)
    assert observation != other_observation

    other_observation = deepcopy(observation)
    _change_agent_orientation(other_observation)
    assert observation != other_observation

    other_observation = deepcopy(observation)
    _change_agent_object(other_observation)
    assert observation != other_observation

    other_observation = deepcopy(observation)
    _change_agent_object(other_observation)
    assert observation != other_observation


def test_observation_hash():
    wall_position = Position(0, 0)
    agent_position = Position(0, 1)
    agent_orientation = Orientation.FORWARD
    agent_object = None

    grid = Grid.from_shape((2, 2))
    grid[wall_position] = Wall()
    agent = Agent(agent_position, agent_orientation, agent_object)
    observation = Observation(grid, agent)

    hash(observation)
