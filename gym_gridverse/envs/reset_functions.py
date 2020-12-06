import itertools as itt
from functools import partial
from typing import Optional, Tuple, Type

import more_itertools as mitt
import numpy as np
import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.design import (
    draw_line_horizontal,
    draw_line_vertical,
    draw_room_grid,
    draw_wall_boundary,
)
from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Floor,
    Goal,
    GridObject,
    Key,
    MovingObstacle,
    Wall,
)
from gym_gridverse.info import Agent, Grid
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State


class ResetFunction(Protocol):
    def __call__(self, *, rng: Optional[rnd.Generator] = None) -> State:
        ...


def reset_minigrid_empty(
    height: int,
    width: int,
    random_agent: bool = False,
    random_goal: bool = False,
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    """imitates Minigrid's Empty environment"""

    if height < 4 or width < 4:
        raise ValueError('height and width need to be at least 4')

    rng = get_gv_rng_if_none(rng)

    # TODO test creation (e.g. count number of walls, goals, check held item)

    grid = Grid(height, width)
    draw_wall_boundary(grid)

    if random_goal:
        goal_y = rng.integers(1, height - 2, endpoint=True)
        goal_x = rng.integers(1, width - 2, endpoint=True)
    else:
        goal_y = height - 2
        goal_x = width - 2

    grid[goal_y, goal_x] = Goal()

    if random_agent:
        agent_position = rng.choice(
            [
                position
                for position in grid.positions()
                if isinstance(grid[position], Floor)
            ]
        )
        agent_orientation = rng.choice(list(Orientation))
    else:
        agent_position = (1, 1)
        agent_orientation = Orientation.E

    agent = Agent(agent_position, agent_orientation)
    return State(grid, agent)


def reset_minigrid_rooms(  # pylint: disable=too-many-locals
    height: int,
    width: int,
    layout: Tuple[int, int],
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:

    rng = get_gv_rng_if_none(rng)

    # TODO test creation (e.g. count number of walls, goals, check held item)

    layout_height, layout_width = layout

    y_splits = np.linspace(
        0,
        height - 1,
        num=layout_height + 1,
        dtype=int,
    )

    if len(y_splits) != len(set(y_splits)):
        raise ValueError(
            f'insufficient height ({height}) for layout ({layout})'
        )

    x_splits = np.linspace(
        0,
        width - 1,
        num=layout_width + 1,
        dtype=int,
    )

    if len(x_splits) != len(set(x_splits)):
        raise ValueError(f'insufficient width ({height}) for layout ({layout})')

    grid = Grid(height, width)
    draw_room_grid(grid, y_splits, x_splits, Wall)

    # passages in horizontal walls
    for y in y_splits[1:-1]:
        for x_from, x_to in mitt.pairwise(x_splits):
            x = rng.integers(x_from + 1, x_to)
            grid[y, x] = Floor()

    # passages in vertical walls
    for y_from, y_to in mitt.pairwise(y_splits):
        for x in x_splits[1:-1]:
            y = rng.integers(y_from + 1, y_to)
            grid[y, x] = Floor()

    # sample agent and goal positions
    agent_position, goal_position = rng.choice(
        [
            position
            for position in grid.positions()
            if isinstance(grid[position], Floor)
        ],
        size=2,
        replace=False,
    )
    agent_orientation = rng.choice(list(Orientation))

    grid[goal_position] = Goal()
    agent = Agent(agent_position, agent_orientation)
    return State(grid, agent)


def reset_minigrid_dynamic_obstacles(
    height: int,
    width: int,
    num_obstacles: int,
    random_agent_pos: bool = False,
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    """Returns an initial state as seen in 'Minigrid-dynamic-obstacle' environment

    Args:
        height (`int`): height of grid
        width (`int`): width of grid
        num_obstacles (`int`): number of dynamic obstacles
        random_agent (`bool, optional`): position of agent, in corner if False
        rng: (`Generator, optional`)

    Returns:
        State:
    """

    rng = get_gv_rng_if_none(rng)

    state = reset_minigrid_empty(height, width, random_agent_pos, rng=rng)
    vacant_positions = [
        position
        for position in state.grid.positions()
        if isinstance(state.grid[position], Floor)
        and position != state.agent.position
    ]

    try:
        sample_positions = rng.choice(
            vacant_positions, size=num_obstacles, replace=False
        )
    except ValueError as e:
        raise ValueError(
            f'Too many obstacles ({num_obstacles}) and not enough '
            f'vacant positions ({len(vacant_positions)})'
        ) from e

    for pos in sample_positions:
        assert isinstance(state.grid[pos], Floor)
        state.grid[pos] = MovingObstacle()

    return state


def reset_minigrid_keydoor(
    height: int, width: int, *, rng: Optional[rnd.Generator] = None
) -> State:
    """Returns a state similar to the gym minigrid 'door & key' environment

    Creates a height x width (including outer walls) grid with a random column
    of walls. The agent and a yellow key are randomly dropped left of the
    column, while the goal is placed in the bottom right. For example::

        #########
        # @#    #
        #  D    #
        #K #   G#
        #########

    Args:
        height (`int`):
        width (`int`):
        rng: (`Generator, optional`)

    Returns:
        State:
    """
    if height < 3 or width < 5 or (height, width) == (3, 5):
        raise ValueError(
            f'Shape must larger than (3, 5), given {(height, width)}'
        )

    rng = get_gv_rng_if_none(rng)

    state = reset_minigrid_empty(height, width)
    assert isinstance(state.grid[height - 2, width - 2], Goal)

    # Generate vertical splitting wall
    x_wall = rng.integers(2, width - 3, endpoint=True)
    line_wall = draw_line_vertical(
        state.grid, range(1, height - 1), x_wall, Wall
    )

    # Place yellow, locked door
    pos_wall = rng.choice(line_wall)
    state.grid[pos_wall] = Door(Door.Status.LOCKED, Colors.YELLOW)

    # Place yellow key left of wall
    # XXX: potential general function
    y_key = rng.integers(1, height - 2, endpoint=True)
    x_key = rng.integers(1, x_wall - 1, endpoint=True)
    state.grid[y_key, x_key] = Key(Colors.YELLOW)

    # Place agent left of wall
    # XXX: potential general function
    y_agent = rng.integers(1, height - 2, endpoint=True)
    x_agent = rng.integers(1, x_wall - 1, endpoint=True)
    state.agent.position = (y_agent, x_agent)  # type: ignore
    state.agent.orientation = rng.choice(list(Orientation))

    return state


def reset_minigrid_crossing(  # pylint: disable=too-many-locals
    height: int,
    width: int,
    num_rivers: int,
    object_type: Type[GridObject],
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    """Returns a state similar to the gym minigrid 'simple/lava crossing' environment

    Creates a height x width (including wall) grid with random rows/columns of
    objects called "rivers". The agent needs to navigate river openings to
    reach the goal.  For example::

        #########
        #@    # #
        #### ####
        #     # #
        ## ######
        #       #
        #     # #
        #     #G#
        #########

    Args:
        height (`int`): odd height of grid
        width (`int`): odd width of grid
        num_rivers (`int`): number of `rivers`
        object_type (`Type[GridObject]`): river's object type
        rng: (`Generator, optional`)

    Returns:
        State:
    """
    if height < 5 or height % 2 == 0:
        raise ValueError(
            f"Minigrid crossing environment height must be odd and >= 5, given {height}"
        )

    if width < 5 or width % 2 == 0:
        raise ValueError(
            f"Minigrid crossing environment width must be odd and >= 5, given {width}"
        )

    if num_rivers < 0:
        raise ValueError(
            f"Minigrid crossing environment number of walls must be >= 0, given {height}"
        )

    rng = get_gv_rng_if_none(rng)

    state = reset_minigrid_empty(height, width)
    assert isinstance(state.grid[height - 2, width - 2], Goal)

    # token `horizontal` and `vertical` objects
    h, v = object(), object()

    # all rivers specified by orientation and position
    rivers = list(
        itt.chain(
            ((h, i) for i in range(2, height - 2, 2)),
            ((v, j) for j in range(2, width - 2, 2)),
        )
    )

    # sample subset of random rivers
    rng.shuffle(rivers)  # NOTE: faster than rng.choice
    rivers = rivers[:num_rivers]

    # create horizontal rivers without crossings
    rivers_h = sorted([pos for direction, pos in rivers if direction is h])
    for y in rivers_h:
        draw_line_horizontal(state.grid, y, range(1, width - 1), object_type)

    # create vertical rivers without crossings
    rivers_v = sorted([pos for direction, pos in rivers if direction is v])
    for x in rivers_v:
        draw_line_vertical(state.grid, range(1, height - 1), x, object_type)

    # sample path to goal
    path = [h] * len(rivers_v) + [v] * len(rivers_h)
    rng.shuffle(path)

    # create crossing
    limits_h = [0] + rivers_h + [height - 1]  # horizontal river boundaries
    limits_v = [0] + rivers_v + [width - 1]  # vertical river boundaries
    room_i, room_j = 0, 0  # coordinates of current "room"
    for step_direction in path:
        if step_direction is h:
            i = rng.integers(limits_h[room_i] + 1, limits_h[room_i + 1])
            j = limits_v[room_j + 1]
            room_j += 1

        elif step_direction is v:
            i = limits_h[room_i + 1]
            j = rng.integers(limits_v[room_j] + 1, limits_v[room_j + 1])
            room_i += 1

        else:
            assert False

        state.grid[i, j] = Floor()

    # Place agent on top left
    state.agent.position = (1, 1)  # type: ignore
    state.agent.orientation = Orientation.E

    return state


def factory(
    name: str,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
    layout: Optional[Tuple[int, int]] = None,
    random_agent_pos: Optional[bool] = None,
    num_obstacles: Optional[int] = None,
    num_rivers: Optional[int] = None,
    object_type: Optional[Type[GridObject]] = None,
) -> ResetFunction:

    if name == 'minigrid_empty':
        if None in [height, width, random_agent_pos]:
            raise ValueError(f'invalid parameters for name `{name}`')

        return partial(
            reset_minigrid_empty,
            height=height,
            width=width,
            random_agent=random_agent_pos,
        )

    if name == 'minigrid_rooms':
        if None in [height, width, layout]:
            raise ValueError(f'invalid parameters for name `{name}`')

        return partial(
            reset_minigrid_rooms, height=height, width=width, layout=layout
        )

    if name == 'minigrid_dynamic_obstacles':
        if None in [height, width, num_obstacles, random_agent_pos]:
            raise ValueError(f'invalid parameters for name `{name}`')

        return partial(
            reset_minigrid_dynamic_obstacles,
            height=height,
            width=width,
            num_obstacles=num_obstacles,
            random_agent_pos=random_agent_pos,
        )

    if name == 'minigrid_keydoor':
        if None in [height, width]:
            raise ValueError(f'invalid parameters for name `{name}`')

        return partial(reset_minigrid_keydoor, height, width)

    if name == 'minigrid_crossing':
        if None in [height, width, num_rivers, object_type]:
            raise ValueError(f'invalid parameters for name `{name}`')

        return partial(
            reset_minigrid_crossing,
            height=height,
            width=width,
            num_rivers=num_rivers,
            object_type=object_type,
        )

    raise ValueError(f'invalid reset function name `{name}`')
