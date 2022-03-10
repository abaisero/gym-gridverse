from typing import List

import pytest

from gym_gridverse.geometry import Area, Orientation, Position, Shape
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import (
    Box,
    Color,
    Exit,
    Floor,
    GridObject,
    Hidden,
    Key,
    Wall,
)


@pytest.mark.parametrize(
    'grid,expected',
    [
        (Grid.from_shape((3, 4)), Shape(3, 4)),
        (Grid.from_shape((4, 3)), Shape(4, 3)),
        (Grid.from_shape((5, 5)), Shape(5, 5)),
    ],
)
def test_grid_shape(grid: Grid, expected: Shape):
    assert grid.shape == expected


@pytest.mark.parametrize(
    'grid,position,expected',
    [
        (Grid.from_shape((3, 4)), Position(0, 0), True),
        (Grid.from_shape((3, 4)), Position(2, 3), True),
        (Grid.from_shape((3, 4)), Position(-1, 0), False),
        (Grid.from_shape((3, 4)), Position(0, -1), False),
        (Grid.from_shape((3, 4)), Position(3, 3), False),
        (Grid.from_shape((3, 4)), Position(2, 4), False),
    ],
)
def test_grid_contains(grid: Grid, position: Position, expected: bool):
    assert grid.area.contains(position) == expected


@pytest.mark.parametrize(
    'grid,expected',
    [
        (Grid.from_shape((3, 4)), 12),
        (Grid.from_shape((4, 5)), 20),
    ],
)
def test_grid_positions_all(grid: Grid, expected: int):
    positions = set(grid.area.positions('all'))
    assert len(positions) == expected

    for position in positions:
        assert grid.area.contains(position)


@pytest.mark.parametrize(
    'grid,expected',
    [
        (Grid.from_shape((3, 4)), 10),
        (Grid.from_shape((4, 5)), 14),
    ],
)
def test_grid_positions_border(grid: Grid, expected: int):
    positions = set(grid.area.positions('border'))
    assert len(positions) == expected

    for position in positions:
        assert grid.area.contains(position)


@pytest.mark.parametrize(
    'grid,expected',
    [
        (Grid.from_shape((3, 4)), 2),
        (Grid.from_shape((4, 5)), 6),
    ],
)
def test_grid_positions_inside(grid: Grid, expected: int):
    positions = set(grid.area.positions('inside'))
    assert len(positions) == expected

    for position in positions:
        assert grid.area.contains(position)


def test_grid_object_types():
    grid = Grid.from_shape((3, 4))

    assert grid.object_types() == set([Floor])

    grid[0, 0] = Wall()
    assert grid.object_types() == set([Floor, Wall])

    grid[0, 0] = Exit()
    assert grid.object_types() == set([Floor, Exit])

    grid[1, 1] = Wall()
    assert grid.object_types() == set([Floor, Exit, Wall])


def test_grid_get_item():
    grid = Grid.from_shape((3, 4))

    pos = Position(0, 0)
    assert isinstance(grid[pos], Floor)
    assert grid[pos] is grid[pos]


def test_grid_set_item():
    grid = Grid.from_shape((3, 4))

    pos = Position(0, 0)
    obj = Floor()

    assert grid[pos] is not obj
    grid[pos] = obj
    assert grid[pos] is obj


def test_grid_swap():
    grid = Grid.from_shape((3, 4))

    # caching positions and objects before swap
    objects_before = {
        position: grid[position] for position in grid.area.positions()
    }

    pos1 = Position(0, 0)
    pos2 = Position(1, 1)
    grid.swap(pos1, pos2)

    # caching positions and objects after swap
    objects_after = {
        position: grid[position] for position in grid.area.positions()
    }

    # testing swapped objects
    assert objects_before[pos1] is objects_after[pos2]
    assert objects_before[pos2] is objects_after[pos1]

    # testing all other objects are the same
    for position in grid.area.positions():
        if position not in (pos1, pos2):
            assert objects_before[position] is objects_after[position]


@pytest.mark.parametrize(
    'area,expected_objects',
    [
        (
            Area((-1, 3), (-1, 4)),
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Wall(), Floor(), Wall(), Floor(), Hidden()],
                [Hidden(), Floor(), Wall(), Floor(), Wall(), Hidden()],
                [Hidden(), Wall(), Floor(), Wall(), Floor(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
            ],
        ),
        (Area((1, 1), (1, 2)), [[Wall(), Floor()]]),
        (
            Area((-1, 1), (-1, 1)),
            [
                [Hidden(), Hidden(), Hidden()],
                [Hidden(), Wall(), Floor()],
                [Hidden(), Floor(), Wall()],
            ],
        ),
        (
            Area((1, 3), (2, 4)),
            [
                [Floor(), Wall(), Hidden()],
                [Wall(), Floor(), Hidden()],
                [Hidden(), Hidden(), Hidden()],
            ],
        ),
    ],
)
def test_grid_subgrid(area: Area, expected_objects: List[List[GridObject]]):
    # checkerboard pattern
    grid = Grid(
        [
            [Wall(), Floor(), Wall(), Floor()],
            [Floor(), Wall(), Floor(), Wall()],
            [Wall(), Floor(), Wall(), Floor()],
        ]
    )

    subgrid = grid.subgrid(area)
    expected = Grid(expected_objects)
    assert subgrid == expected


def test_grid_equality():
    """A simple test that equality is not limited to just checking (first) objects"""

    grid_1 = Grid(
        [
            [Wall(), Floor(), Wall(), Floor()],
            [Floor(), Wall(), Floor(), Wall()],
            [Wall(), Floor(), Wall(), Floor()],
        ]
    )

    grid_2 = Grid(
        [
            [Wall(), Floor(), Wall(), Floor()],
            [Floor(), Wall(), Floor(), Wall()],
            [Wall(), Floor(), Wall(), Floor()],
            [Wall(), Floor(), Wall(), Floor()],
        ]
    )

    grid_3 = Grid(
        [
            [Wall(), Floor(), Wall(), Floor(), Floor()],
            [Floor(), Wall(), Floor(), Wall(), Floor()],
            [Wall(), Floor(), Wall(), Floor(), Floor()],
        ]
    )

    assert grid_1 != grid_2
    assert grid_2 != grid_1
    assert grid_1 != grid_3
    assert grid_3 != grid_1
    assert grid_2 != grid_3
    assert grid_3 != grid_2


@pytest.mark.parametrize(
    'objects',
    [
        [
            [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
            [Hidden(), Wall(), Floor(), Wall(), Floor(), Hidden()],
            [Hidden(), Floor(), Wall(), Floor(), Wall(), Hidden()],
            [Hidden(), Wall(), Floor(), Wall(), Floor(), Hidden()],
            [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
        ],
        [[Wall(), Floor()]],
        [
            [Hidden(), Hidden(), Hidden()],
            [Hidden(), Wall(), Floor()],
            [Hidden(), Floor(), Wall()],
        ],
        [
            [Floor(), Wall(), Hidden()],
            [Wall(), Floor(), Hidden()],
            [Hidden(), Hidden(), Hidden()],
        ],
    ],
)
def test_grid_objects(objects):
    assert Grid(objects).objects == objects


def test_grid_subgrid_references():
    key = Key(Color.RED)
    box = Box(key)

    # weird scenario where the key is both in the box and outside the box,
    # only created to test references
    grid = Grid([[key, box]])

    subgrid = grid.subgrid(Area((0, 0), (0, 1)))
    key = subgrid[0, 0]
    box = subgrid[0, 1]
    assert box.content is key


@pytest.mark.parametrize(
    'orientation,expected_objects',
    [
        (
            Orientation.F,
            [
                [Wall(), Floor(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall()],
                [Wall(), Floor(), Wall(), Floor()],
            ],
        ),
        (
            Orientation.B,
            [
                [Floor(), Wall(), Floor(), Wall()],
                [Wall(), Floor(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall()],
            ],
        ),
        (
            Orientation.R,
            [
                [Floor(), Wall(), Floor()],
                [Wall(), Floor(), Wall()],
                [Floor(), Wall(), Floor()],
                [Wall(), Floor(), Wall()],
            ],
        ),
        (
            Orientation.L,
            [
                [Wall(), Floor(), Wall()],
                [Floor(), Wall(), Floor()],
                [Wall(), Floor(), Wall()],
                [Floor(), Wall(), Floor()],
            ],
        ),
    ],
)
def test_grid_mul(
    orientation: Orientation, expected_objects: List[List[GridObject]]
):
    # checkerboard pattern
    grid = Grid(
        [
            [Wall(), Floor(), Wall(), Floor()],
            [Floor(), Wall(), Floor(), Wall()],
            [Wall(), Floor(), Wall(), Floor()],
        ]
    )

    expected = Grid(expected_objects)
    assert grid * orientation == expected
