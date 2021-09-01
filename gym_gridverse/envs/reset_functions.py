import itertools as itt
from functools import partial
from typing import Optional, Set, Tuple, Type

import more_itertools as mitt
import numpy as np
import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.agent import Agent
from gym_gridverse.debugging import checkraise
from gym_gridverse.design import (
    draw_area,
    draw_line_horizontal,
    draw_line_vertical,
    draw_room_grid,
    draw_wall_boundary,
)
from gym_gridverse.geometry import Orientation
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import (
    Beacon,
    Color,
    Door,
    Exit,
    Floor,
    GridObject,
    Key,
    MovingObstacle,
    Telepod,
    Wall,
)
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.state import State


class ResetFunction(Protocol):
    def __call__(self, *, rng: Optional[rnd.Generator] = None) -> State:
        ...


def reset_empty(
    height: int,
    width: int,
    random_agent: bool = False,
    random_exit: bool = False,
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    """An empty environment"""

    checkraise(
        lambda: height >= 4 and width >= 4,
        ValueError,
        'height and width need to be at least 4',
    )

    rng = get_gv_rng_if_none(rng)

    # TODO: test creation (e.g. count number of walls, exits, check held item)

    grid = Grid(height, width)
    draw_wall_boundary(grid)

    if random_exit:
        exit_y = rng.integers(1, height - 2, endpoint=True)
        exit_x = rng.integers(1, width - 2, endpoint=True)
    else:
        exit_y = height - 2
        exit_x = width - 2

    grid[exit_y, exit_x] = Exit()

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


def reset_rooms(  # pylint: disable=too-many-locals
    height: int,
    width: int,
    layout: Tuple[int, int],
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:

    rng = get_gv_rng_if_none(rng)

    # TODO: test creation (e.g. count number of walls, exits, check held item)

    layout_height, layout_width = layout

    y_splits = np.linspace(
        0,
        height - 1,
        num=layout_height + 1,
        dtype=int,
    )

    checkraise(
        lambda: len(y_splits) == len(set(y_splits)),
        ValueError,
        'insufficient height ({}) for layout ({})',
        height,
        layout,
    )

    x_splits = np.linspace(
        0,
        width - 1,
        num=layout_width + 1,
        dtype=int,
    )

    checkraise(
        lambda: len(x_splits) == len(set(x_splits)),
        ValueError,
        'insufficient width ({}) for layout ({})',
        height,
        layout,
    )

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

    # sample agent and exit positions
    agent_position, exit_position = rng.choice(
        [
            position
            for position in grid.positions()
            if isinstance(grid[position], Floor)
        ],
        size=2,
        replace=False,
    )
    agent_orientation = rng.choice(list(Orientation))

    grid[exit_position] = Exit()
    agent = Agent(agent_position, agent_orientation)
    return State(grid, agent)


def reset_dynamic_obstacles(
    height: int,
    width: int,
    num_obstacles: int,
    random_agent_pos: bool = False,
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    """An environment with dynamically moving obstacles

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

    state = reset_empty(height, width, random_agent_pos, rng=rng)
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


def reset_keydoor(
    height: int, width: int, *, rng: Optional[rnd.Generator] = None
) -> State:
    """An environment with a key and a door

    Creates a height x width (including outer walls) grid with a random column
    of walls. The agent and a yellow key are randomly dropped left of the
    column, while the exit is placed in the bottom right. For example::

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
    checkraise(
        lambda: height >= 3 and width >= 5 and (height, width) != (3, 5),
        ValueError,
        'Shape must larger than (3, 5), given {}',
        (height, width),
    )

    rng = get_gv_rng_if_none(rng)

    state = reset_empty(height, width)
    assert isinstance(state.grid[height - 2, width - 2], Exit)

    # Generate vertical splitting wall
    x_wall = rng.integers(2, width - 3, endpoint=True)
    line_wall = draw_line_vertical(
        state.grid, range(1, height - 1), x_wall, Wall
    )

    # Place yellow, locked door
    pos_wall = rng.choice(line_wall)
    state.grid[pos_wall] = Door(Door.Status.LOCKED, Color.YELLOW)

    # Place yellow key left of wall
    # XXX: potential general function
    y_key = rng.integers(1, height - 2, endpoint=True)
    x_key = rng.integers(1, x_wall - 1, endpoint=True)
    state.grid[y_key, x_key] = Key(Color.YELLOW)

    # Place agent left of wall
    # XXX: potential general function
    y_agent = rng.integers(1, height - 2, endpoint=True)
    x_agent = rng.integers(1, x_wall - 1, endpoint=True)
    state.agent.position = (y_agent, x_agent)  # type: ignore
    state.agent.orientation = rng.choice(list(Orientation))

    return state


def reset_crossing(  # pylint: disable=too-many-locals
    height: int,
    width: int,
    num_rivers: int,
    object_type: Type[GridObject],
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    """An environment with "rivers" to be crosses

    Creates a height x width (including wall) grid with random rows/columns of
    objects called "rivers". The agent needs to navigate river openings to
    reach the exit.  For example::

        #########
        #@    # #
        #### ####
        #     # #
        ## ######
        #       #
        #     # #
        #     #E#
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
    checkraise(
        lambda: height >= 5 and height % 2 == 1,
        ValueError,
        'Crossing environment height must be odd and >= 5, given {}',
        height,
    )
    checkraise(
        lambda: width >= 5 and width % 2 == 1,
        ValueError,
        'Crossing environment width must be odd and >= 5, given {}',
        width,
    )
    checkraise(
        lambda: num_rivers >= 0,
        ValueError,
        'Crossing environment number of walls must be >= 0, given {}',
        num_rivers,
    )

    rng = get_gv_rng_if_none(rng)

    state = reset_empty(height, width)
    assert isinstance(state.grid[height - 2, width - 2], Exit)

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

    # sample path to exit
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


def reset_teleport(
    height: int, width: int, *, rng: Optional[rnd.Generator] = None
) -> State:

    rng = get_gv_rng_if_none(rng)

    state = reset_empty(height, width)
    assert isinstance(state.grid[height - 2, width - 2], Exit)

    # Place agent on top left
    state.agent.position = (1, 1)  # type: ignore
    state.agent.orientation = rng.choice([Orientation.E, Orientation.S])

    num_telepods = 2
    telepods = [Telepod(Color.RED) for _ in range(num_telepods)]
    positions = rng.choice(
        [
            position
            for position in state.grid.positions()
            if isinstance(state.grid[position], Floor)
            and position != state.agent.position
        ],
        size=num_telepods,
        replace=False,
    )
    for position, telepod in zip(positions, telepods):
        state.grid[position] = telepod

    # Place agent on top left
    state.agent.position = (1, 1)  # type: ignore
    state.agent.orientation = rng.choice([Orientation.E, Orientation.S])

    return state


def reset_memory(
    height: int,
    width: int,
    colors: Set[Color],
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:

    checkraise(
        lambda: height >= 5,
        ValueError,
        'Memory environment height must be >= 5, given {}',
        height,
    )
    checkraise(
        lambda: width >= 5 and width % 2 == 1,
        ValueError,
        'Memory environment height must be odd and >= 5, given {}',
        height,
    )
    checkraise(
        lambda: Color.NONE not in colors,
        ValueError,
        'Memory environment colors must not include NONE, given {}',
        colors,
    )
    checkraise(
        lambda: len(colors) >= 2,
        ValueError,
        'Memory environment colors must have at least 2 colors, given {}',
        colors,
    )

    rng = get_gv_rng_if_none(rng)

    grid = Grid(height, width)
    draw_area(grid, grid.area, Wall, fill=True)
    draw_line_horizontal(grid, 1, range(2, width - 2), Floor)
    draw_line_horizontal(grid, height - 2, range(2, width - 2), Floor)
    draw_line_vertical(grid, range(2, height - 2), width // 2, Floor)

    color_good, color_bad = rng.choice(list(colors), size=2, replace=False)
    x_exit_good, x_exit_bad = rng.choice([1, width - 2], size=2, replace=False)
    grid[1, x_exit_good] = Exit(color_good)
    grid[1, x_exit_bad] = Exit(color_bad)
    grid[height - 2, 1] = Beacon(color_good)
    grid[height - 2, width - 2] = Beacon(color_good)

    agent_position = (height // 2, width // 2)
    agent_orientation = Orientation.N
    agent = Agent(agent_position, agent_orientation)

    return State(grid, agent)


def reset_memory_rooms(
    height: int,
    width: int,
    layout: Tuple[int, int],
    colors: Set[Color],
    num_beacons: int,
    num_exits: int,
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:

    checkraise(
        lambda: Color.NONE not in colors,
        ValueError,
        'Memory-rooms environment colors must not include NONE, given {}',
        colors,
    )
    checkraise(
        lambda: len(colors) >= 2,
        ValueError,
        'Memory-rooms environment colors must have at least 2 colors, given {}',
        colors,
    )
    checkraise(
        lambda: num_beacons >= 1,
        ValueError,
        'Memory-rooms environment must have at least 1 beacon, given {}',
        num_beacons,
    )
    checkraise(
        lambda: num_exits >= 2,
        ValueError,
        'Memory-rooms environment must have at least 2 exit, given {}',
        num_exits,
    )

    rng = get_gv_rng_if_none(rng)

    layout_height, layout_width = layout

    y_splits = np.linspace(
        0,
        height - 1,
        num=layout_height + 1,
        dtype=int,
    )

    checkraise(
        lambda: len(y_splits) == len(set(y_splits)),
        ValueError,
        'insufficient height ({}) for layout ({})',
        height,
        layout,
    )

    x_splits = np.linspace(
        0,
        width - 1,
        num=layout_width + 1,
        dtype=int,
    )

    checkraise(
        lambda: len(x_splits) == len(set(x_splits)),
        ValueError,
        'insufficient width ({}) for layout ({})',
        height,
        layout,
    )

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

    # sample agent, beacon, and exit positions
    positions = rng.choice(
        [
            position
            for position in grid.positions()
            if isinstance(grid[position], Floor)
        ],
        size=1 + num_beacons + num_exits,
        replace=False,
    )

    agent_position = positions[0]
    agent_orientation = rng.choice(list(Orientation))
    agent = Agent(agent_position, agent_orientation)

    sample_colors = rng.choice(list(colors), size=num_exits, replace=False)

    good_color = sample_colors[0]
    beacon_positions = positions[1 : 1 + num_beacons]
    for beacon_position in beacon_positions:
        grid[beacon_position] = Beacon(good_color)

    exit_positions = positions[1 + num_beacons :]
    for exit_position, exit_color in zip(exit_positions, sample_colors):
        grid[exit_position] = Exit(exit_color)

    return State(grid, agent)


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
    colors: Optional[Set[Color]] = None,
    num_beacons: Optional[int] = None,
    num_exits: Optional[int] = None,
) -> ResetFunction:

    if name == 'empty':
        checkraise(
            lambda: height is not None
            and width is not None
            and random_agent_pos is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(
            reset_empty,
            height=height,
            width=width,
            random_agent=random_agent_pos,
        )

    if name == 'rooms':
        checkraise(
            lambda: height is not None
            and width is not None
            and layout is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(reset_rooms, height=height, width=width, layout=layout)

    if name == 'dynamic_obstacles':
        checkraise(
            lambda: height is not None
            and width is not None
            and num_obstacles is not None
            and random_agent_pos is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(
            reset_dynamic_obstacles,
            height=height,
            width=width,
            num_obstacles=num_obstacles,
            random_agent_pos=random_agent_pos,
        )

    if name == 'keydoor':
        checkraise(
            lambda: height is not None and width is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(reset_keydoor, height, width)

    if name == 'crossing':
        checkraise(
            lambda: height is not None
            and width is not None
            and num_rivers is not None
            and object_type is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(
            reset_crossing,
            height=height,
            width=width,
            num_rivers=num_rivers,
            object_type=object_type,
        )

    if name == 'teleport':
        checkraise(
            lambda: height is not None and width is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(reset_teleport, height=height, width=width)

    if name == 'memory':
        checkraise(
            lambda: height is not None
            and width is not None
            and colors is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(reset_memory, height=height, width=width, colors=colors)

    if name == 'memory_rooms':
        checkraise(
            lambda: height is not None
            and width is not None
            and layout is not None
            and colors is not None
            and num_beacons is not None
            and num_exits is not None,
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return partial(
            reset_memory_rooms,
            height=height,
            width=width,
            layout=layout,
            colors=colors,
            num_beacons=num_beacons,
            num_exits=num_exits,
        )

    raise ValueError(f'invalid reset function name `{name}`')
