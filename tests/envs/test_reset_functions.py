from typing import Type

import pytest

from gym_gridverse.envs.reset_functions import (
    crossing,
    dynamic_obstacles,
    factory,
    keydoor,
    teleport,
)
from gym_gridverse.geometry import Shape
from gym_gridverse.grid_object import (
    Door,
    Exit,
    Key,
    MovingObstacle,
    Telepod,
    Wall,
)
from gym_gridverse.state import State


def wall_column(size: int, state: State) -> int:
    for x in range(1, size - 1):
        if isinstance(state.grid[1, x], (Wall, Door)):
            return x

    raise ValueError('Could not find wall column in state')


@pytest.mark.parametrize(
    'size', [-5, 4]  # negative size and positive-but-too-small
)
def test_reset_keydoor_throw_if_too_small(size: int):
    """Asserts method throws if provided size is too small"""
    with pytest.raises(ValueError):
        keydoor(Shape(size, size))


@pytest.mark.parametrize('size', range(5, 12))
def test_reset_keydoor_wall(size: int):
    """Tests whether the reset state contains a wall column"""
    state = keydoor(Shape(size, size))

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
def test_reset_keydoor_agent_is_left_of_wall(size: int):
    state = keydoor(Shape(size, size))
    assert state.agent.position.x < wall_column(
        size, state
    ), "Agent should be left of wall"


@pytest.mark.parametrize('size', range(5, 12))
def test_reset_keydoor_key(size: int):
    state = keydoor(Shape(size, size))

    key_pos = [
        pos
        for pos in state.grid.area.positions()
        if isinstance(state.grid[pos], Key)
    ]

    assert len(key_pos) == 1, "There should be exactly 1 key"
    assert key_pos[0].x < wall_column(size, state), "Key should be left of wall"


@pytest.mark.parametrize('size', range(5, 12))
def test_reset_keydoor_exit(size: int):
    state = keydoor(Shape(size, size))
    assert isinstance(
        state.grid[size - 2, size - 2], Exit
    ), "There should be a exit bottom right"


@pytest.mark.parametrize('height', [10, 20])
@pytest.mark.parametrize('width', [10, 20])
@pytest.mark.parametrize('num_obstacles', [5, 10])
def test_reset_dynamic_obstacles(height: int, width: int, num_obstacles: int):
    state = dynamic_obstacles(Shape(height, width), num_obstacles)
    assert state.grid.shape == Shape(height, width)

    state_num_obstacles = sum(
        isinstance(state.grid[position], MovingObstacle)
        for position in state.grid.area.positions()
    )
    assert state_num_obstacles == num_obstacles


@pytest.mark.parametrize('height', [9, 13])
@pytest.mark.parametrize('width', [9, 13])
@pytest.mark.parametrize('num_rivers', [3, 5])
def test_reset_crossing(height: int, width: int, num_rivers: int):
    state = crossing(Shape(height, width), num_rivers, object_type=Wall)
    assert state.grid.shape == Shape(height, width)


@pytest.mark.parametrize('height', [9, 13])
@pytest.mark.parametrize('width', [9, 13])
def test_reset_teleport(height: int, width: int):
    state = teleport(Shape(height, width))
    assert state.grid.shape == Shape(height, width)

    num_telepods = sum(
        isinstance(state.grid[pos], Telepod)
        for pos in state.grid.area.positions()
    )
    assert num_telepods == 2


@pytest.mark.parametrize(
    'name,kwargs',
    [
        (
            'empty',
            {
                'shape': Shape(10, 10),
                'random_agent_pos': True,
            },
        ),
        (
            'rooms',
            {
                'shape': Shape(10, 10),
                'layout': (2, 2),
            },
        ),
        (
            'dynamic_obstacles',
            {
                'shape': Shape(10, 10),
                'num_obstacles': 10,
                'random_agent_pos': True,
            },
        ),
        (
            'keydoor',
            {
                'shape': Shape(10, 10),
            },
        ),
        (
            'crossing',
            {
                'shape': Shape(9, 9),
                'num_rivers': 3,
                'object_type': Wall,
            },
        ),
    ],
)
def test_factory_valid(name: str, kwargs):
    factory(name, **kwargs)


@pytest.mark.parametrize(
    'name,kwargs,exception',
    [
        ('invalid', {}, ValueError),
        ('empty', {}, ValueError),
        ('rooms', {}, ValueError),
        ('dynamic_obstacles', {}, ValueError),
        ('keydoor', {}, ValueError),
        ('crossing', {}, ValueError),
    ],
)
def test_factory_invalid(name: str, kwargs, exception: Type[Exception]):
    with pytest.raises(exception):
        factory(name, **kwargs)
