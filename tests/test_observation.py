import dataclasses
from copy import deepcopy

import pytest

from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import Color, Floor, Key, NoneGridObject, Wall
from gym_gridverse.info import Agent, Grid
from gym_gridverse.observation import Observation


def _change_grid(observation: Observation):
    """changes one object in the grid"""
    observation.grid[0, 0] = (
        Wall() if isinstance(observation.grid[0, 0], Floor) else Floor()
    )


def _change_agent_position(observation: Observation):
    """changes agent position"""
    observation.agent.position = dataclasses.replace(
        observation.agent.position,
        y=(observation.agent.position.y + 1) % observation.grid.height,
        x=(observation.agent.position.x + 1) % observation.grid.width,
    )


def _change_agent_orientation(observation: Observation):
    """changes agent orientation"""
    observation.agent.orientation = observation.agent.orientation.rotate_back()


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
        Observation(Grid(2, 3), Agent((0, 0), Orientation.N)),
        Observation(Grid(3, 2), Agent((1, 1), Orientation.S, Key(Color.RED))),
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
    wall_position = (0, 0)
    agent_position = (0, 1)
    agent_orientation = Orientation.N
    agent_object = None

    grid = Grid(2, 2)
    grid[wall_position] = Wall()
    agent = Agent(agent_position, agent_orientation, agent_object)
    observation = Observation(grid, agent)

    hash(observation)
