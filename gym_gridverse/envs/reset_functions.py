import inspect
import itertools as itt
import warnings
from functools import partial
from typing import List, Optional, Set, Tuple, Type

import more_itertools as mitt
import numpy as np
import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.agent import Agent
from gym_gridverse.design import (
    draw_area,
    draw_line_horizontal,
    draw_line_vertical,
    draw_room_grid,
    draw_wall_boundary,
)
from gym_gridverse.geometry import Orientation, Position, Shape
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
from gym_gridverse.rng import choice, choices, get_gv_rng_if_none, shuffle
from gym_gridverse.state import State
from gym_gridverse.utils.custom import import_if_custom
from gym_gridverse.utils.functions import checkraise_kwargs, select_kwargs
from gym_gridverse.utils.protocols import get_keyword_parameter
from gym_gridverse.utils.registry import FunctionRegistry


class ResetFunction(Protocol):
    """Signature that all reset functions must follow"""

    def __call__(self, *, rng: Optional[rnd.Generator] = None) -> State:
        ...


class ResetFunctionRegistry(FunctionRegistry):
    def get_protocol_parameters(
        self, signature: inspect.Signature
    ) -> List[inspect.Parameter]:
        rng = get_keyword_parameter(signature, 'rng')
        return [rng]

    def check_signature(self, function: ResetFunction):
        signature = inspect.signature(function)
        (rng,) = self.get_protocol_parameters(signature)

        # checks `rng` is keyword
        if rng.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ]:
            raise TypeError(
                f'The `rng` argument ({rng.name}) '
                f'of a registered reward function ({function}) '
                'should be allowed to be a keyword argument.'
            )

        # checks if annotations, if given, are consistent
        if rng.annotation not in [
            inspect.Parameter.empty,
            Optional[rnd.Generator],
        ]:
            warnings.warn(
                f'The `rng` argument ({rng.name}) '
                f'of a registered reward function ({function}) '
                f'has an annotation ({rng.annotation}) '
                'which is not `Optional[rnd.Generator]`.'
            )

        if signature.return_annotation not in [inspect.Parameter.empty, State]:
            warnings.warn(
                f'The return type of a registered reset function ({function}) '
                f'has an annotation ({signature.return_annotation}) '
                'which is not `State`.'
            )


reset_function_registry = ResetFunctionRegistry()
"""Reset function registry"""


@reset_function_registry.register
def empty(
    shape: Shape,
    random_agent: bool = False,
    random_exit: bool = False,
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    """An empty environment"""
    # TODO: test

    if shape.height < 4 or shape.width < 4:
        raise ValueError('height and width need to be at least 4')

    rng = get_gv_rng_if_none(rng)

    # TODO: test creation (e.g. count number of walls, exits, check held item)

    grid = Grid.from_shape((shape.height, shape.width))
    draw_wall_boundary(grid)

    exit_y: int
    exit_x: int

    if random_exit:
        exit_y = rng.integers(1, shape.height - 2, endpoint=True)
        exit_x = rng.integers(1, shape.width - 2, endpoint=True)
    else:
        exit_y = shape.height - 2
        exit_x = shape.width - 2

    grid[exit_y, exit_x] = Exit()

    if random_agent:
        positions = [
            position
            for position in grid.area.positions()
            if isinstance(grid[position], Floor)
        ]
        agent_position = choice(rng, positions)
        agent_orientation = choice(rng, list(Orientation))
    else:
        agent_position = Position(1, 1)
        agent_orientation = Orientation.R

    agent = Agent(agent_position, agent_orientation)
    return State(grid, agent)


@reset_function_registry.register
def rooms(
    shape: Shape,
    layout: Tuple[int, int],
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    rng = get_gv_rng_if_none(rng)

    # TODO: test creation (e.g. count number of walls, exits, check held item)

    layout_height, layout_width = layout

    y_splits = np.linspace(
        0,
        shape.height - 1,
        num=layout_height + 1,
        dtype=int,
    )

    if len(y_splits) != len(set(y_splits)):
        raise ValueError(
            f'insufficient height ({shape.height}) for layout ({layout})'
        )

    x_splits = np.linspace(
        0,
        shape.width - 1,
        num=layout_width + 1,
        dtype=int,
    )

    if len(x_splits) != len(set(x_splits)):
        raise ValueError(
            f'insufficient width ({shape.width}) for layout ({layout})'
        )

    grid = Grid.from_shape((shape.height, shape.width))
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
    positions = [
        position
        for position in grid.area.positions()
        if isinstance(grid[position], Floor)
    ]
    agent_position, exit_position = choices(
        rng,
        positions,
        size=2,
        replace=False,
    )
    agent_orientation = choice(rng, list(Orientation))

    grid[exit_position] = Exit()
    agent = Agent(agent_position, agent_orientation)
    return State(grid, agent)


@reset_function_registry.register
def dynamic_obstacles(
    shape: Shape,
    num_obstacles: int,
    random_agent: bool = False,
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    """An environment with dynamically moving obstacles

    Args:
        shape (`Shape`): shape of grid
        num_obstacles (`int`): number of dynamic obstacles
        random_agent (`bool, optional`): position of agent, in corner if False
        rng: (`Generator, optional`)

    Returns:
        State:
    """

    rng = get_gv_rng_if_none(rng)

    state = empty(shape, random_agent, rng=rng)
    vacant_positions = [
        position
        for position in state.grid.area.positions()
        if isinstance(state.grid[position], Floor)
        and position != state.agent.position
    ]

    try:
        sample_positions = choices(
            rng, vacant_positions, size=num_obstacles, replace=False
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


@reset_function_registry.register
def keydoor(shape: Shape, *, rng: Optional[rnd.Generator] = None) -> State:
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
        shape (`Shape`):
        rng: (`Generator, optional`)

    Returns:
        State:
    """

    if shape.height < 3 or shape.width < 5 or shape == Shape(3, 5):
        raise ValueError(f'Shape must larger than (3, 5), given {shape}')

    rng = get_gv_rng_if_none(rng)

    state = empty(shape)
    assert isinstance(state.grid[shape.height - 2, shape.width - 2], Exit)

    # Generate vertical splitting wall
    x_wall = rng.integers(2, shape.width - 3, endpoint=True)
    line_wall = draw_line_vertical(
        state.grid, range(1, shape.height - 1), x_wall, Wall
    )

    # Place yellow, locked door
    pos_wall = choice(rng, line_wall)
    state.grid[pos_wall] = Door(Door.Status.LOCKED, Color.YELLOW)

    # Place yellow key left of wall
    # XXX: potential general function
    y_key = rng.integers(1, shape.height - 2, endpoint=True)
    x_key = rng.integers(1, x_wall - 1, endpoint=True)
    state.grid[y_key, x_key] = Key(Color.YELLOW)

    # Place agent left of wall
    # XXX: potential general function
    y_agent = rng.integers(1, shape.height - 2, endpoint=True)
    x_agent = rng.integers(1, x_wall - 1, endpoint=True)
    state.agent.position = Position(y_agent, x_agent)
    state.agent.orientation = choice(rng, list(Orientation))

    return state


@reset_function_registry.register
def crossing(
    shape: Shape,
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
        shape (`Shape`): shape (odd height and width) of grid
        num_rivers (`int`): number of `rivers`
        object_type (`Type[GridObject]`): river's object type
        rng: (`Generator, optional`)

    Returns:
        State:
    """

    if shape.height < 5 or shape.height % 2 == 0:
        raise ValueError(f'height ({shape.height}) must be odd and >= 5')
    if shape.width < 5 or shape.width % 2 == 0:
        raise ValueError(f'width ({shape.width}) must be odd and >= 5')
    if num_rivers <= 0:
        raise ValueError(f'number of rivers ({num_rivers}) must be positive')

    rng = get_gv_rng_if_none(rng)

    state = empty(shape)
    assert isinstance(state.grid[shape.height - 2, shape.width - 2], Exit)

    # token `horizontal` and `vertical` objects
    h, v = object(), object()

    # all rivers specified by orientation and position
    rivers = list(
        itt.chain(
            ((h, i) for i in range(2, shape.height - 2, 2)),
            ((v, j) for j in range(2, shape.width - 2, 2)),
        )
    )

    # sample subset of random rivers
    rivers = shuffle(rng, rivers)
    rivers = rivers[:num_rivers]

    # create horizontal rivers without crossings
    rivers_h = sorted([pos for direction, pos in rivers if direction is h])
    for y in rivers_h:
        draw_line_horizontal(
            state.grid, y, range(1, shape.width - 1), object_type
        )

    # create vertical rivers without crossings
    rivers_v = sorted([pos for direction, pos in rivers if direction is v])
    for x in rivers_v:
        draw_line_vertical(
            state.grid, range(1, shape.height - 1), x, object_type
        )

    # sample path to exit
    path = [h] * len(rivers_v) + [v] * len(rivers_h)
    path = shuffle(rng, path)

    # create crossing
    limits_h = (
        [0] + rivers_h + [shape.height - 1]
    )  # horizontal river boundaries
    limits_v = [0] + rivers_v + [shape.width - 1]  # vertical river boundaries
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
    state.agent.position = Position(1, 1)
    state.agent.orientation = Orientation.R

    return state


@reset_function_registry.register
def teleport(shape: Shape, *, rng: Optional[rnd.Generator] = None) -> State:
    rng = get_gv_rng_if_none(rng)

    state = empty(shape)
    assert isinstance(state.grid[shape.height - 2, shape.width - 2], Exit)

    # Place agent on top left
    state.agent.position = Position(1, 1)
    state.agent.orientation = choice(rng, [Orientation.R, Orientation.B])

    num_telepods = 2
    telepods = [Telepod(Color.RED) for _ in range(num_telepods)]
    positions = rng.choice(
        [
            position
            for position in state.grid.area.positions()
            if isinstance(state.grid[position], Floor)
            and position != state.agent.position
        ],
        size=num_telepods,
        replace=False,
    )
    for position, telepod in zip(positions, telepods):
        state.grid[position] = telepod

    # Place agent on top left
    state.agent.position = Position(1, 1)
    state.agent.orientation = choice(rng, [Orientation.R, Orientation.B])

    return state


@reset_function_registry.register
def memory(
    shape: Shape,
    colors: Set[Color],
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    if shape.height < 5:
        raise ValueError(f'height ({shape.height}) must be >= 5')
    if shape.width < 5 or shape.width % 2 == 0:
        raise ValueError(f'width ({shape.width}) must be odd and >= 5')
    if Color.NONE in colors:
        raise ValueError(f'colors ({colors}) must not include Colors.NONE')
    if len(colors) < 2:
        raise ValueError(f'colors ({colors}) must have at least 2 colors')

    rng = get_gv_rng_if_none(rng)

    grid = Grid.from_shape((shape.height, shape.width))
    draw_area(grid, grid.area, Wall, fill=True)
    draw_line_horizontal(grid, 1, range(2, shape.width - 2), Floor)
    draw_line_horizontal(
        grid, shape.height - 2, range(2, shape.width - 2), Floor
    )
    draw_line_vertical(
        grid, range(2, shape.height - 2), shape.width // 2, Floor
    )

    color_good, color_bad = choices(rng, list(colors), size=2, replace=False)
    x_exit_good, x_exit_bad = choices(
        rng, [1, shape.width - 2], size=2, replace=False
    )
    grid[1, x_exit_good] = Exit(color_good)
    grid[1, x_exit_bad] = Exit(color_bad)
    grid[shape.height - 2, 1] = Beacon(color_good)
    grid[shape.height - 2, shape.width - 2] = Beacon(color_good)

    agent_position = Position(shape.height // 2, shape.width // 2)
    agent_orientation = Orientation.F
    agent = Agent(agent_position, agent_orientation)

    return State(grid, agent)


@reset_function_registry.register
def memory_rooms(
    shape: Shape,
    layout: Tuple[int, int],
    colors: Set[Color],
    num_beacons: int,
    num_exits: int,
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    if Color.NONE in colors:
        raise ValueError(f'colors ({colors}) must not include NONE')
    if len(colors) < 2:
        raise ValueError(f'colors ({colors}) must have at least 2 colors')
    if num_beacons < 1:
        raise ValueError(f'num_beacons ({num_beacons}) must be positive')
    if num_exits < 2:
        raise ValueError(f'num_exits ({num_exits}) must be >= 2')

    rng = get_gv_rng_if_none(rng)

    layout_height, layout_width = layout

    y_splits = np.linspace(
        0,
        shape.height - 1,
        num=layout_height + 1,
        dtype=int,
    )

    if len(y_splits) != len(set(y_splits)):
        raise ValueError(
            f'insufficient shape.height ({shape.height}) for layout ({layout})'
        )

    x_splits = np.linspace(
        0,
        shape.width - 1,
        num=layout_width + 1,
        dtype=int,
    )

    if len(x_splits) != len(set(x_splits)):
        raise ValueError(
            f'insufficient shape.width ({shape.width}) for layout ({layout})'
        )

    grid = Grid.from_shape((shape.height, shape.width))
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
    positions = [
        position
        for position in grid.area.positions()
        if isinstance(grid[position], Floor)
    ]
    positions = choices(
        rng,
        positions,
        size=1 + num_beacons + num_exits,
        replace=False,
    )

    agent_position = positions[0]
    agent_orientation = choice(rng, list(Orientation))
    agent = Agent(agent_position, agent_orientation)

    sample_colors = choices(rng, list(colors), size=num_exits, replace=False)

    good_color = sample_colors[0]
    beacon_positions = positions[1 : 1 + num_beacons]
    for beacon_position in beacon_positions:
        grid[beacon_position] = Beacon(good_color)

    exit_positions = positions[1 + num_beacons :]
    for exit_position, exit_color in zip(exit_positions, sample_colors):
        grid[exit_position] = Exit(exit_color)

    return State(grid, agent)


def factory(name: str, **kwargs) -> ResetFunction:
    name = import_if_custom(name)

    try:
        function = reset_function_registry[name]
    except KeyError as error:
        raise ValueError(f'invalid reset function name {name}') from error

    signature = inspect.signature(function)
    required_keys = [
        parameter.name
        for parameter in reset_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is inspect.Parameter.empty
    ]
    optional_keys = [
        parameter.name
        for parameter in reset_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is not inspect.Parameter.empty
    ]

    checkraise_kwargs(kwargs, required_keys)
    kwargs = select_kwargs(kwargs, required_keys + optional_keys)
    return partial(function, **kwargs)
