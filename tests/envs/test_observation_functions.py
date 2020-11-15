from typing import Sequence
from unittest.mock import MagicMock

import pytest

from gym_gridverse.envs.observation_functions import (
    factory,
    minigrid_observation,
)
from gym_gridverse.geometry import Orientation, Shape
from gym_gridverse.grid_object import Floor, GridObject, Hidden, Wall
from gym_gridverse.info import Agent, Grid
from gym_gridverse.spaces import ObservationSpace
from gym_gridverse.state import State


@pytest.mark.parametrize(
    'agent',
    [
        Agent((7, 7), Orientation.N),
        Agent((3, 3), Orientation.S),
        Agent((7, 3), Orientation.E),
        Agent((3, 7), Orientation.W),
    ],
)
def test_minigrid_observation(agent: Agent):
    grid = Grid(10, 10)
    grid[5, 5] = Wall()

    state = State(grid, agent)
    observation_space = ObservationSpace(Shape(6, 5), [], [])
    observation = minigrid_observation(
        state, observation_space=observation_space
    )
    assert observation.agent.position == (5, 2)
    assert observation.agent.orientation == Orientation.N
    assert observation.agent.obj == state.agent.obj
    assert observation.grid.shape == (6, 5)
    assert isinstance(observation.grid[3, 0], Wall)


@pytest.mark.parametrize(
    'agent,expected_objects',
    [
        (
            Agent((2, 1), Orientation.N),
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Wall(), Wall(), Wall(), Hidden(),],
                [Hidden(), Floor(), Floor(), Floor(), Hidden(),],
            ],
        ),
        (
            Agent((0, 1), Orientation.S),
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Wall(), Wall(), Wall(), Hidden(),],
                [Hidden(), Floor(), Floor(), Floor(), Hidden(),],
            ],
        ),
        (
            Agent((2, 1), Orientation.E),
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Wall(), Floor(), Hidden(), Hidden(),],
                [Hidden(), Wall(), Floor(), Hidden(), Hidden(),],
            ],
        ),
        (
            Agent((2, 1), Orientation.W),
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(),],
                [Hidden(), Hidden(), Floor(), Wall(), Hidden(),],
                [Hidden(), Hidden(), Floor(), Wall(), Hidden(),],
            ],
        ),
    ],
)
def test_minigrid_observation_partially_observable(
    agent: Agent, expected_objects: Sequence[Sequence[GridObject]],
):
    grid = Grid.from_objects(
        [
            [Floor(), Floor(), Floor()],
            [Wall(), Wall(), Wall()],
            [Floor(), Floor(), Floor()],
        ]
    )
    state = State(grid, agent)
    observation_space = ObservationSpace(Shape(6, 5), [], [])
    observation = minigrid_observation(
        state, observation_space=observation_space
    )
    expected = Grid.from_objects(expected_objects)
    assert observation.grid == expected


@pytest.mark.parametrize(
    'name,kwargs',
    [
        ('full_visibility', {}),
        ('from_visibility', {'visibility_function': MagicMock()}),
        ('minigrid_observation', {}),
        ('raytracing_observation', {}),
        ('stochastic_raytracing_observation', {}),
    ],
)
def test_factory_valid(name: str, kwargs):
    observation_space = MagicMock()
    factory(name, observation_space=observation_space, **kwargs)


@pytest.mark.parametrize(
    'name,kwargs,exception',
    [
        ('invalid', {}, ValueError),
        ('full_visibility', {}, ValueError),
        ('from_visibility', {}, ValueError),
        ('minigrid_observation', {}, ValueError),
        ('raytracing_observation', {}, ValueError),
        ('stochastic_raytracing_observation', {}, ValueError),
    ],
)
def test_factory_invalid(name: str, kwargs, exception: Exception):
    with pytest.raises(exception):  # type: ignore
        factory(name, **kwargs)
