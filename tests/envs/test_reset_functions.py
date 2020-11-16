import pytest

from gym_gridverse.envs.reset_functions import (
    factory,
    reset_minigrid_door_key,
    reset_minigrid_dynamic_obstacles,
)
from gym_gridverse.geometry import Shape
from gym_gridverse.grid_object import Door, Goal, Key, MovingObstacle, Wall
from gym_gridverse.state import State


def wall_column(size: int, state: State) -> int:
    for x in range(1, size - 1):
        if isinstance(state.grid[1, x], (Wall, Door)):
            return x

    raise ValueError('Could not find wall column in state')


@pytest.mark.parametrize(
    'size', [-5, 4]  # negative size and positive-but-too-small
)
def test_reset_minigrid_door_key_throw_if_too_small(size: int):
    """Asserts method throws if provided size is too small"""
    with pytest.raises(ValueError):
        reset_minigrid_door_key(size)


@pytest.mark.parametrize('size', range(5, 12))
def test_reset_minigrid_door_key_wall(size: int):
    """Tests whether the reset state contains a wall column"""
    state = reset_minigrid_door_key(size)

    # Surrounded by walls
    for i in range(0, size):
        assert isinstance(state.grid[0, i], Wall)
        assert isinstance(state.grid[i, 0], Wall)

    count = 0
    for x in range(1, size - 1):
        if isinstance(state.grid[1, x], (Wall, Door)):
            count += 1
            column = x
    assert count == 1, "Should only be one column of walls"
    assert wall_column(size, state) == column

    count = 0
    for y in range(1, size - 1):
        if isinstance(state.grid[y, column], Door):
            count += 1
        else:
            assert isinstance(state.grid[y, column], Wall)
    assert count == 1, "There should be exactly 1 door"


@pytest.mark.parametrize('size', range(5, 12))
def test_reset_minigrid_door_key_agent_is_left_of_wall(size: int):
    state = reset_minigrid_door_key(size)
    assert state.agent.position.x < wall_column(
        size, state
    ), "Agent should be left of wall"


@pytest.mark.parametrize('size', range(5, 12))
def test_reset_minigrid_door_key_key(size: int):
    state = reset_minigrid_door_key(size)

    key_pos = [
        pos
        for pos in state.grid.positions()
        if isinstance(state.grid[pos], Key)
    ]

    assert len(key_pos) == 1, "There should be exactly 1 key"
    assert key_pos[0].x < wall_column(size, state), "Key should be left of wall"


@pytest.mark.parametrize('size', range(5, 12))
def test_reset_minigrid_door_key_goal(size: int):
    state = reset_minigrid_door_key(size)
    assert isinstance(
        state.grid[size - 2, size - 2], Goal
    ), "There should be a goal bottom right"


@pytest.mark.parametrize('height', [10, 20])
@pytest.mark.parametrize('width', [10, 20])
@pytest.mark.parametrize('num_obstacles', [5, 10])
def test_reset_minigrid_dynamic_obstacles_num_obstacles_grid_size(
    height: int, width: int, num_obstacles: int
):
    state = reset_minigrid_dynamic_obstacles(height, width, num_obstacles)
    assert state.grid.shape == Shape(height, width)


@pytest.mark.parametrize('height', [10, 20])
@pytest.mark.parametrize('width', [10, 20])
@pytest.mark.parametrize('num_obstacles', [5, 10])
def test_reset_minigrid_dynamic_obstacles_num_obstacles(
    height: int, width: int, num_obstacles: int
):
    state = reset_minigrid_dynamic_obstacles(height, width, num_obstacles)
    state_num_obstacles = sum(
        isinstance(state.grid[position], MovingObstacle)
        for position in state.grid.positions()
    )
    assert state_num_obstacles == num_obstacles


@pytest.mark.parametrize(
    'name,kwargs',
    [
        (
            'minigrid_empty',
            {'height': 10, 'width': 10, 'random_agent_pos': True},
        ),
        ('minigrid_four_rooms', {'height': 10, 'width': 10}),
        (
            'minigrid_dynamic_obstacles',
            {
                'height': 10,
                'width': 10,
                'num_obstacles': 10,
                'random_agent_pos': True,
            },
        ),
        ('minigrid_door_key', {'size': 10}),
    ],
)
def test_factory_valid(name: str, kwargs):
    factory(name, **kwargs)


@pytest.mark.parametrize(
    'name,kwargs,exception',
    [
        ('invalid', {}, ValueError),
        ('minigrid_empty', {}, ValueError),
        ('minigrid_four_rooms', {}, ValueError),
        ('minigrid_dynamic_obstacles', {}, ValueError),
        ('minigrid_door_key', {}, ValueError),
    ],
)
def test_factory_invalid(name: str, kwargs, exception: Exception):
    with pytest.raises(exception):  # type: ignore
        factory(name, **kwargs)
