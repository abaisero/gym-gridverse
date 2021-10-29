from typing import Iterable, List

from gym_gridverse.geometry import Area, Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import GridObjectFactory, Wall


def draw_wall_boundary(grid: Grid) -> List[Position]:
    """draw boundary of walls on grid"""
    return draw_room(grid, grid.area, Wall)


def draw_room(
    grid: Grid, area: Area, factory: GridObjectFactory
) -> List[Position]:
    """use factory-created grid-objects to draw room boundary on grid"""
    return draw_area(grid, area, factory, fill=False)


def draw_room_grid(
    grid: Grid, ys: Iterable[int], xs: Iterable[int], factory: GridObjectFactory
) -> List[Position]:
    """use factory-created grid-objects to draw a grid of rooms on grid"""
    y_range = range(min(ys), max(ys) + 1)
    x_range = range(min(xs), max(xs) + 1)

    # draw horizontal lines
    positions = draw_cartesian_product(grid, ys, x_range, factory)

    # fill in remaining vertical lines
    ys_remaining = [y for y in y_range if y not in ys]
    positions += draw_cartesian_product(grid, ys_remaining, xs, factory)

    return positions


def draw_area(
    grid: Grid, area: Area, factory: GridObjectFactory, *, fill: bool
) -> List[Position]:
    """use factory-created grid-objects to draw area on grid"""
    positions = list(area.positions('all' if fill else 'border'))
    for pos in positions:
        grid[pos] = factory()

    return positions


def draw_line_horizontal(
    grid: Grid, y: int, xs: Iterable[int], factory: GridObjectFactory
) -> List[Position]:
    """use factory-created grid-objects to draw horizontal line on grid"""
    return draw_cartesian_product(grid, [y], xs, factory)


def draw_line_vertical(
    grid: Grid, ys: Iterable[int], x: int, factory: GridObjectFactory
) -> List[Position]:
    """use factory-created grid-objects to draw vertical line on grid"""
    return draw_cartesian_product(grid, ys, [x], factory)


def draw_cartesian_product(
    grid: Grid, ys: Iterable[int], xs: Iterable[int], factory: GridObjectFactory
) -> List[Position]:
    """use factory-created grid-objects to draw on grid"""
    positions = [Position(y, x) for y in ys for x in xs]
    for position in positions:
        grid[position] = factory()

    return positions
