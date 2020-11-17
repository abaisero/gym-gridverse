from functools import partial
from typing import Callable, Optional

from gym_gridverse.geometry import Orientation
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Floor,
    Goal,
    Key,
    MovingObstacle,
    Wall,
)
from gym_gridverse.info import Agent, Grid
from gym_gridverse.rng import get_gv_rng
from gym_gridverse.state import State

ResetFunction = Callable[[], State]


def reset_minigrid_empty(
    height: int, width: int, random_agent: bool = False
) -> State:
    """imitates Minigrid's Empty environment"""

    if height < 4 or width < 4:
        raise ValueError('height and width need to be at least 4')

    rng = get_gv_rng()

    # TODO test creation (e.g. count number of walls, goals, check held item)

    objects = []
    objects.append([Wall() for x in range(width)])
    for _ in range(1, height - 2):
        objects.append(
            [Wall()] + [Floor() for _ in range(1, width - 1)] + [Wall()]  # type: ignore
        )

    objects.append(
        [Wall()] + [Floor() for _ in range(1, width - 2)] + [Goal(), Wall()]  # type: ignore
    )
    objects.append([Wall() for x in range(width)])
    grid = Grid.from_objects(objects)

    if random_agent:
        agent_position = rng.choice(
            [
                position
                for position in grid.positions()
                if isinstance(grid[position], Floor)
            ]
        )
        agent_orientation = rng.choice(list(Orientation))
        agent = Agent(agent_position, agent_orientation)
    else:
        agent = Agent((1, 1), Orientation.E)

    return State(grid, agent)


def reset_minigrid_four_rooms(height: int, width: int) -> State:
    """imitates Minigrid's FourRooms environment"""
    if height < 5 or width < 5:
        raise ValueError('height and width need to be at least 5')

    rng = get_gv_rng()

    # TODO test creation (e.g. count number of walls, goals, check held item)

    height_split = height // 2
    width_split = width // 2

    grid = Grid(height, width)

    # making walls
    for y in range(height):
        grid[y, 0] = Wall()
        grid[y, width_split] = Wall()
        grid[y, width - 1] = Wall()

    for x in range(width):
        grid[0, x] = Wall()
        grid[height_split, x] = Wall()
        grid[height - 1, x] = Wall()

    # creating openings in walls
    y = rng.choice(range(1, height_split - 1))
    grid[y, width_split] = Floor()
    y = rng.choice(range(height_split + 1, height - 1))
    grid[y, width_split] = Floor()

    x = rng.choice(range(1, width_split - 1))
    grid[height_split, x] = Floor()
    x = rng.choice(range(width_split + 1, width - 1))
    grid[height_split, x] = Floor()

    # random goal
    goal_position = rng.choice(
        [
            position
            for position in grid.positions()
            if isinstance(grid[position], Floor)
        ]
    )
    grid[goal_position] = Goal()

    # random agent
    agent_position = rng.choice(
        [
            position
            for position in grid.positions()
            if isinstance(grid[position], Floor)
        ]
    )
    agent_orientation = rng.choice(list(Orientation))

    agent = Agent(agent_position, agent_orientation)
    return State(grid, agent)


def reset_minigrid_dynamic_obstacles(
    height: int, width: int, num_obstacles: int, random_agent_pos: bool = False
) -> State:
    """Returns an initial state as seen in 'Minigrid-dynamic-obstacle' environment

    Args:
        height (`int`): height of grid
        width (`int`): width of grid
        num_obstacles (`int`): number of dynamic obstacles
        random_agent (`bool, optional`): position of agent, in corner if False

    Returns:
        State:
    """

    rng = get_gv_rng()

    state = reset_minigrid_empty(height, width, random_agent_pos)
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


def reset_minigrid_door_key(grid_size: int) -> State:
    """Returns a state similar to the gym minigrid 'door & key' environment

    Creates a grid_size x grid_size (including wall) grid with a random column
    of walls. The agent and a yellow key are randomly dropped left of the
    column, while the goal is placed in the bottom right. For example::

        #########
        # @#    #
        #  D    #
        #K #   G#
        #########

    Args:
        grid_size (`int`): assumes rectangular grid

    Returns:
        State:
    """
    if grid_size < 5:
        raise ValueError(
            f"Minigrid door-key environment minimum size is 5, given {grid_size}"
        )

    rng = get_gv_rng()

    state = reset_minigrid_empty(grid_size, grid_size)
    assert isinstance(state.grid[grid_size - 2, grid_size - 2], Goal)

    # Generate vertical splitting wall
    wall_column = rng.integers(2, grid_size - 3, endpoint=True)
    # XXX: potential general function
    for h in range(0, grid_size):
        state.grid[h, wall_column] = Wall()

    # Place yellow, locked door
    door_row = rng.integers(2, grid_size - 2, endpoint=True)
    state.grid[door_row, wall_column] = Door(Door.Status.LOCKED, Colors.YELLOW)

    # Place yellow key left of wall
    # XXX: potential general function
    y = rng.integers(1, grid_size - 2, endpoint=True)
    x = rng.integers(1, wall_column - 1, endpoint=True)
    key_pos = (y, x)
    state.grid[key_pos] = Key(Colors.YELLOW)

    # Place agent left of wall
    # XXX: potential general function
    y = rng.integers(1, grid_size - 2, endpoint=True)
    x = rng.integers(1, wall_column - 1, endpoint=True)
    state.agent.position = (y, x)  # type: ignore
    state.agent.orientation = rng.choice(list(Orientation))

    return state


def factory(
    name: str,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
    size: Optional[int] = None,
    random_agent_pos: Optional[bool] = None,
    num_obstacles: Optional[int] = None,
) -> ResetFunction:

    if name == 'minigrid_empty':
        if None in [height, width, random_agent_pos]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            reset_minigrid_empty,
            height=height,
            width=width,
            random_agent=random_agent_pos,
        )

    if name == 'minigrid_four_rooms':
        if None in [height, width]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(reset_minigrid_four_rooms, height=height, width=width)

    if name == 'minigrid_dynamic_obstacles':
        if None in [height, width, num_obstacles, random_agent_pos]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(
            reset_minigrid_dynamic_obstacles,
            height=height,
            width=width,
            num_obstacles=num_obstacles,
            random_agent_pos=random_agent_pos,
        )

    if name == 'minigrid_door_key':
        if None in [size]:
            raise ValueError('invalid parameters for name `{name}`')

        return partial(reset_minigrid_door_key, grid_size=size)

    raise ValueError(f'invalid reset function name `{name}`')
