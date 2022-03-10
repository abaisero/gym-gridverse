from typing import List, Type
from unittest.mock import MagicMock

import pytest

from gym_gridverse.agent import Agent
from gym_gridverse.envs.observation_functions import factory, partially_occluded
from gym_gridverse.geometry import Orientation, Position, Shape
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Floor, GridObject, Hidden, Wall
from gym_gridverse.spaces import ObservationSpace
from gym_gridverse.state import State


@pytest.mark.parametrize(
    'agent',
    [
        Agent(Position(7, 7), Orientation.F),
        Agent(Position(3, 3), Orientation.B),
        Agent(Position(7, 3), Orientation.R),
        Agent(Position(3, 7), Orientation.L),
    ],
)
def test_partially_occluded_1(agent: Agent):
    grid = Grid.from_shape((10, 10))
    grid[5, 5] = Wall()

    state = State(grid, agent)
    observation_space = ObservationSpace(Shape(6, 5), [], [])
    observation = partially_occluded(state, area=observation_space.area)
    assert observation.agent.position == Position(5, 2)
    assert observation.agent.orientation == Orientation.F
    assert observation.agent.grid_object == state.agent.grid_object
    assert observation.grid.shape == Shape(6, 5)
    assert isinstance(observation.grid[3, 0], Wall)


@pytest.mark.parametrize(
    'agent,expected_objects',
    [
        (
            Agent(Position(2, 1), Orientation.F),
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Wall(), Wall(), Wall(), Hidden()],
                [Hidden(), Floor(), Floor(), Floor(), Hidden()],
            ],
        ),
        (
            Agent(Position(0, 1), Orientation.B),
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Wall(), Wall(), Wall(), Hidden()],
                [Hidden(), Floor(), Floor(), Floor(), Hidden()],
            ],
        ),
        (
            Agent(Position(2, 1), Orientation.R),
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Wall(), Floor(), Hidden(), Hidden()],
                [Hidden(), Wall(), Floor(), Hidden(), Hidden()],
            ],
        ),
        (
            Agent(Position(2, 1), Orientation.L),
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Hidden(), Floor(), Wall(), Hidden()],
                [Hidden(), Hidden(), Floor(), Wall(), Hidden()],
            ],
        ),
    ],
)
def test_artially_occluded_2(
    agent: Agent, expected_objects: List[List[GridObject]]
):
    grid = Grid(
        [
            [Floor(), Floor(), Floor()],
            [Wall(), Wall(), Wall()],
            [Floor(), Floor(), Floor()],
        ]
    )
    state = State(grid, agent)
    observation_space = ObservationSpace(Shape(6, 5), [], [])
    observation = partially_occluded(state, area=observation_space.area)
    expected = Grid(expected_objects)
    assert observation.grid == expected


@pytest.mark.parametrize(
    'name,kwargs',
    [
        ('from_visibility', {'visibility_function': MagicMock()}),
        ('fully_transparent', {}),
        ('partially_occluded', {}),
        ('raytracing', {}),
        ('stochastic_raytracing', {}),
    ],
)
def test_factory_valid(name: str, kwargs):
    observation_space = MagicMock()
    factory(name, area=observation_space.area, **kwargs)


@pytest.mark.parametrize(
    'name,kwargs,exception',
    [
        ('invalid', {}, ValueError),
        ('from_visibility', {}, ValueError),
        ('fully_transparent', {}, ValueError),
        ('partially_occluded', {}, ValueError),
        ('raytracing', {}, ValueError),
        ('stochastic_raytracing', {}, ValueError),
    ],
)
def test_factory_invalid(name: str, kwargs, exception: Type[Exception]):
    with pytest.raises(exception):
        factory(name, **kwargs)
