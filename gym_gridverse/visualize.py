""" Visualization of Gridverse
* Attempt to make it as little invasive as possible
* Assumes grid objects can be represented with a single char string
"""

from typing import List

from termcolor import colored

from gym_gridverse.grid_object import Colors, GridObject, Wall
from gym_gridverse.state import Grid, Orientation, State

COLORS_TO_TERMCOLOR = {
    Colors.NONE: 'black',
    Colors.RED: 'red',
    Colors.GREEN: 'green',
    Colors.BLUE: 'blue',
    Colors.YELLOW: 'yellow',
}

ORIENTATION_TO_STR = {
    Orientation.N: "^",
    Orientation.S: "v",
    Orientation.E: ">",
    Orientation.W: "<",
}


def str_render_agent(orientation: Orientation) -> str:
    return colored(ORIENTATION_TO_STR[orientation], attrs=["bold"])


def str_render_object(obj: GridObject) -> str:
    if obj.color == Colors.NONE:
        return obj.render_as_char()
    return colored(obj.render_as_char(), COLORS_TO_TERMCOLOR[obj.color])


def str_render_grid(grid: Grid) -> List[List[str]]:

    string_grid: List[List[str]] = [
        ["" for _ in range(grid.width)] for _ in range(grid.height)
    ]

    h = -1
    for i, pos in enumerate(grid.positions()):

        w = i % grid.width
        if w == 0:
            h += 1

        string_grid[h][w] = str_render_object(grid[pos])

    return string_grid


def str_wall_row(grid_width: int) -> str:
    """Returns string representation of a row of wall

    Args:
        grid_width (`int`): width of grid
    """
    return str_render_object(Wall()) * (grid_width + 2)


def str_render_state(state: State) -> str:

    string_grid = str_render_grid(state.grid)
    agent_x, agent_y = state.agent.position

    string_grid[agent_x][agent_y] = str_render_agent(state.agent.orientation)

    grid_descr = "\n".join(
        [
            str_render_object(Wall()) + ''.join(row) + str_render_object(Wall())
            for row in string_grid
        ]
    )

    grid_descr = (
        str_wall_row(state.grid.width)
        + "\n"
        + grid_descr
        + "\n"
        + str_wall_row(state.grid.width)
    )

    if not state.agent.obj:
        return grid_descr

    return grid_descr + "\nAgent holding: " + str_render_object(state.agent.obj)
