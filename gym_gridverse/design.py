from typing import List, Tuple

from gym_gridverse.geometry import Area, Position
from gym_gridverse.grid_object import GridObjectFactory
from gym_gridverse.info import Grid


def draw_room(
    grid: Grid, area: Area, factory: GridObjectFactory
) -> List[Position]:
    """use factory-created grid-objects to draw room boundary on grid"""
    return draw_area(grid, area, factory, fill=False)


def draw_area(
    grid: Grid, area: Area, factory: GridObjectFactory, *, fill
) -> List[Position]:
    """use factory-created grid-objects to draw area on grid"""
    positions = list(area.positions() if fill else area.positions_border())
    for pos in positions:
        grid[pos] = factory()

    return positions


def draw_line_horizontal(
    grid: Grid, y: int, xs: Tuple[int, int], factory: GridObjectFactory
):
    """use factory-created grid-objects to draw horizontal line on grid"""
    xmin, xmax = min(xs), max(xs)
    positions = [Position(y, x) for x in range(xmin, xmax + 1)]
    for pos in positions:
        grid[pos] = factory()

    return positions


def draw_line_vertical(
    grid: Grid, ys: Tuple[int, int], x: int, factory: GridObjectFactory
) -> List[Position]:
    """use factory-created grid-objects to draw vertical line on grid"""
    ymin, ymax = min(ys), max(ys)
    positions = [Position(y, x) for y in range(ymin, ymax + 1)]
    for pos in positions:
        grid[pos] = factory()

    return positions
