import random
from typing import Callable

from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid_object import Floor, Goal, Wall
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State

ResetFunction = Callable[[], State]


def reset_minigrid_empty(
    height: int, width: int, random_agent: bool = False
) -> State:
    """imitates Minigrid's Empty environment"""
    if height < 4 or width < 4:
        raise ValueError('height and width need to be at least 4')

    # TODO test creation (e.g. count number of walls, goals, check held item)

    objects = []
    objects.append([Wall() for x in range(width)])
    for _ in range(1, height - 2):
        objects.append(
            [Wall()] + [Floor() for _ in range(1, width - 1)] + [Wall()]
        )
    objects.append(
        [Wall()] + [Floor() for _ in range(1, width - 2)] + [Goal(), Wall()]
    )
    objects.append([Wall() for x in range(width)])
    grid = Grid.from_objects(objects)

    if random_agent:
        agent_position = random.choice(
            [
                position
                for position in grid.positions()
                if isinstance(grid[position], Floor)
            ]
        )
        agent_orientation = random.choice(list(Orientation))
    else:
        agent_position = Position(1, 1)
        agent_orientation = Orientation.E

    agent = Agent(agent_position, agent_orientation)
    return State(grid, agent)


def reset_minigrid_four_rooms(height: int, width: int) -> State:
    """imitates Minigrid's FourRooms environment"""
    if height < 5 or width < 5:
        raise ValueError('height and width need to be at least 5')

    # TODO test creation (e.g. count number of walls, goals, check held item)

    height_split = height // 2
    width_split = width // 2

    grid = Grid(height, width)

    # making walls
    for y in range(height):
        grid[Position(y, 0)] = Wall()
        grid[Position(y, width_split)] = Wall()
        grid[Position(y, width - 1)] = Wall()

    for x in range(width):
        grid[Position(0, x)] = Wall()
        grid[Position(height_split, x)] = Wall()
        grid[Position(height - 1, x)] = Wall()

    # creating openings in walls
    y = random.choice(range(1, height_split - 1))
    grid[Position(y, width_split)] = Floor()
    y = random.choice(range(height_split + 1, height - 1))
    grid[Position(y, width_split)] = Floor()

    x = random.choice(range(1, width_split - 1))
    grid[Position(height_split, x)] = Floor()
    x = random.choice(range(width_split + 1, width - 1))
    grid[Position(height_split, x)] = Floor()

    # random goal
    goal_position = random.choice(
        [
            position
            for position in grid.positions()
            if isinstance(grid[position], Floor)
        ]
    )
    grid[goal_position] = Goal()

    # random agent
    agent_position = random.choice(
        [
            position
            for position in grid.positions()
            if isinstance(grid[position], Floor)
        ]
    )
    agent_orientation = random.choice(list(Orientation))

    agent = Agent(agent_position, agent_orientation)
    return State(grid, agent)
