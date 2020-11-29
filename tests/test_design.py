import pytest

from gym_gridverse.design import (
    draw_area,
    draw_line_horizontal,
    draw_line_vertical,
    draw_room,
)
from gym_gridverse.geometry import Area, Range
from gym_gridverse.grid_object import Wall
from gym_gridverse.info import Grid


@pytest.mark.parametrize(
    'grid,area,expected',
    [
        (Grid(4, 5), Area((0, 3), (0, 4)), 14),
        (Grid(4, 5), Area((1, 2), (1, 3)), 6),
    ],
)
def test_draw_room(grid: Grid, area: Area, expected: int):
    positions = draw_room(grid, area, Wall)
    assert len(positions) == len(set(positions))

    num_walls = sum(isinstance(grid[pos], Wall) for pos in grid.positions())
    assert len(positions) == num_walls == expected


@pytest.mark.parametrize(
    'grid,area,fill,expected',
    [
        (Grid(4, 5), Area((0, 3), (0, 4)), True, 20),
        (Grid(4, 5), Area((0, 3), (0, 4)), False, 14),
        (Grid(4, 5), Area((1, 2), (1, 3)), True, 6),
        (Grid(4, 5), Area((1, 2), (1, 3)), False, 6),
    ],
)
def test_draw_area(grid: Grid, area: Area, fill: bool, expected: int):
    positions = draw_area(grid, area, Wall, fill=fill)
    assert len(positions) == len(set(positions))

    num_walls = sum(isinstance(grid[pos], Wall) for pos in grid.positions())
    assert len(positions) == num_walls == expected


@pytest.mark.parametrize(
    'grid,y,xs,expected',
    [
        (Grid(4, 5), 1, (0, 4), 5),
        (Grid(4, 5), 1, (1, 3), 3),
    ],
)
def test_draw_line_horizontal(grid: Grid, y: int, xs: Range, expected: int):
    positions = draw_line_horizontal(grid, y, xs, Wall)
    assert len(positions) == len(set(positions))

    num_walls = sum(isinstance(grid[pos], Wall) for pos in grid.positions())
    assert len(positions) == num_walls == expected


@pytest.mark.parametrize(
    'grid,ys,x,expected',
    [
        (Grid(4, 5), (0, 3), 1, 4),
        (Grid(4, 5), (1, 2), 1, 2),
    ],
)
def test_draw_line_vertical(grid: Grid, ys: Range, x: int, expected: int):
    positions = draw_line_vertical(grid, ys, x, Wall)
    assert len(positions) == len(set(positions))

    num_walls = sum(isinstance(grid[pos], Wall) for pos in grid.positions())
    assert len(positions) == num_walls == expected
