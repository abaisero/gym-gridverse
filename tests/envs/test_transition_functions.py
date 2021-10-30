""" Tests state dynamics """

import copy
from typing import Sequence
from unittest.mock import MagicMock, Mock, patch

import pytest

from gym_gridverse.action import Action
from gym_gridverse.agent import Agent
from gym_gridverse.envs.reset_functions import dynamic_obstacles
from gym_gridverse.envs.transition_functions import (
    actuate_box,
    actuate_door,
    factory,
    move_agent,
    move_obstacles,
    pickndrop,
    teleport,
    transition_with_copy,
    turn_agent,
)
from gym_gridverse.geometry import Orientation, Position, Shape
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import (
    Box,
    Color,
    Door,
    Floor,
    GridObject,
    Key,
    MovingObstacle,
    NoneGridObject,
    Telepod,
    Wall,
)
from gym_gridverse.state import State


def make_moving_obstacle_state():
    grid = Grid.from_shape((3, 3))
    grid[1, 1] = MovingObstacle()
    agent = MagicMock()
    return State(grid, agent)


@pytest.mark.parametrize(
    'orientation,actions,expected',
    [
        # Facing north, rotate to WEST
        (
            Orientation.N,
            [Action.TURN_LEFT],
            Orientation.W,
        ),
        # Not rotating
        (
            Orientation.W,
            [Action.MOVE_LEFT, Action.ACTUATE, Action.PICK_N_DROP],
            Orientation.W,
        ),
        # Two rotation to EAST
        (
            Orientation.W,
            [Action.TURN_LEFT, Action.TURN_LEFT],
            Orientation.E,
        ),
        # One back to SOUTH
        (
            Orientation.E,
            [Action.TURN_RIGHT],
            Orientation.S,
        ),
        # Full circle for fun to SOUTH
        (
            Orientation.S,
            [
                Action.TURN_RIGHT,
                Action.TURN_RIGHT,
                Action.TURN_RIGHT,
                Action.TURN_RIGHT,
            ],
            Orientation.S,
        ),
    ],
)
def test_turn_agent(
    orientation: Orientation,
    actions: Sequence[Action],
    expected: Orientation,
):

    state = Mock(agent=Mock(orientation=orientation))
    for action in actions:
        turn_agent(state, action)

    assert state.agent.orientation == expected


@pytest.mark.parametrize(
    'position,orientation,actions,expected',
    [
        #  unblocked movement
        (Position(2, 1), Orientation.N, [Action.MOVE_LEFT], Position(2, 0)),
        (Position(2, 0), Orientation.N, [Action.MOVE_RIGHT], Position(2, 1)),
        (
            Position(2, 1),
            Orientation.N,
            [Action.MOVE_FORWARD, Action.MOVE_FORWARD],
            Position(0, 1),
        ),
        (Position(0, 1), Orientation.N, [Action.MOVE_BACKWARD], Position(1, 1)),
        # blocked by grid object
        # blocked by edges
        (Position(2, 1), Orientation.N, [Action.MOVE_RIGHT], Position(2, 1)),
        # non-movements
        (Position(2, 1), Orientation.N, [Action.TURN_RIGHT], Position(2, 1)),
        (Position(2, 1), Orientation.N, [Action.TURN_LEFT], Position(2, 1)),
        (Position(2, 1), Orientation.N, [Action.ACTUATE], Position(2, 1)),
        (Position(2, 1), Orientation.N, [Action.PICK_N_DROP], Position(2, 1)),
        # facing east
        (
            Position(2, 1),
            Orientation.E,
            [Action.MOVE_LEFT, Action.MOVE_LEFT],
            Position(0, 1),
        ),
        (Position(0, 1), Orientation.E, [Action.MOVE_BACKWARD], Position(0, 0)),
        (Position(0, 0), Orientation.E, [Action.MOVE_FORWARD], Position(0, 1)),
        (Position(0, 1), Orientation.E, [Action.MOVE_FORWARD], Position(0, 1)),
        (Position(0, 1), Orientation.E, [Action.MOVE_RIGHT], Position(1, 1)),
    ],
)
def test_move_agent(
    position: Position,
    orientation: Orientation,
    actions: Sequence[Action],
    expected: Position,
):
    state = State(
        Grid.from_shape((3, 2)),
        Agent(position, orientation),
    )

    for action in actions:
        move_agent(state, action)

    assert state.agent.position == expected


# TODO: integrate with previous test
def test_move_agent_blocked_by_grid_object():
    """Puts an object on (2,0) and try to move there"""
    state = State(
        Grid.from_shape((3, 2)),
        Agent(Position(2, 1), Orientation.N),
    )

    state.grid[2, 0] = Door(Door.Status.CLOSED, Color.YELLOW)
    move_agent(state, Action.MOVE_LEFT)

    assert state.agent.position == Position(2, 1)


# TODO: integrate with previous test
def test_move_agent_can_go_on_non_block_objects():
    state = State(
        Grid.from_shape((3, 2)),
        Agent(Position(2, 1), Orientation.N),
    )

    state.grid[2, 0] = Door(Door.Status.OPEN, Color.YELLOW)
    move_agent(state, Action.MOVE_LEFT)
    assert state.agent.position == Position(2, 0)

    state.grid[2, 1] = Key(Color.BLUE)
    move_agent(state, Action.MOVE_RIGHT)
    assert state.agent.position == Position(2, 1)


def test_pickup_mechanics_nothing_to_pickup():
    grid = Grid.from_shape((3, 4))
    agent = Agent(Position(1, 2), Orientation.S)
    item_position = Position(2, 2)

    state = State(grid, agent)

    # Cannot pickup floor
    next_state = transition_with_copy(pickndrop, state, Action.PICK_N_DROP)
    assert state == next_state

    # Cannot pickup door
    grid[item_position] = Door(Door.Status.CLOSED, Color.GREEN)
    next_state = transition_with_copy(pickndrop, state, Action.PICK_N_DROP)
    assert state == next_state

    assert isinstance(next_state.grid[item_position], Door)


def test_pickup_mechanics_pickup():
    grid = Grid.from_shape((3, 4))
    agent = Agent(Position(1, 2), Orientation.S)
    item_position = Position(2, 2)

    grid[item_position] = Key(Color.GREEN)
    state = State(grid, agent)

    # Pick up works
    next_state = transition_with_copy(pickndrop, state, Action.PICK_N_DROP)
    assert grid[item_position] == next_state.agent.obj
    assert isinstance(next_state.grid[item_position], Floor)

    # Pick up only works with correct action
    next_state = transition_with_copy(pickndrop, state, Action.MOVE_LEFT)
    assert grid[item_position] != next_state.agent.obj
    assert next_state.grid[item_position] == grid[item_position]


def test_pickup_mechanics_drop():
    grid = Grid.from_shape((3, 4))
    agent = Agent(Position(1, 2), Orientation.S)
    item_position = Position(2, 2)

    agent.obj = Key(Color.BLUE)
    state = State(grid, agent)

    # Can drop:
    next_state = transition_with_copy(pickndrop, state, Action.PICK_N_DROP)
    assert isinstance(next_state.agent.obj, NoneGridObject)
    assert agent.obj == next_state.grid[item_position]

    # Cannot drop:
    state.grid[item_position] = Wall()

    next_state = transition_with_copy(pickndrop, state, Action.PICK_N_DROP)
    assert isinstance(next_state.grid[item_position], Wall)
    assert agent.obj == next_state.agent.obj


def test_pickup_mechanics_swap():
    grid = Grid.from_shape((3, 4))
    agent = Agent(Position(1, 2), Orientation.S)
    item_position = Position(2, 2)

    agent.obj = Key(Color.BLUE)
    grid[item_position] = Key(Color.GREEN)
    state = State(grid, agent)

    next_state = transition_with_copy(pickndrop, state, Action.PICK_N_DROP)
    assert state.grid[item_position] == next_state.agent.obj
    assert state.agent.obj == next_state.grid[item_position]


# Tests each moving obstacle is moved exactly once.  A previous naive
# implementation simply iterated over all positions and moved the
# encountered MovingObstacle objects.  If the obstacle were moved to a
# later position, it would eventually be moved again at least once.  This
# test is here to make sure that does not happen again.
def test_move_obstacles_once_per_obstacle():
    state = dynamic_obstacles(Shape(6, 6), num_obstacles=4)

    moved_obstacles = []

    # wrapper swap method to collect swapped GridObjects
    def swap_patch(p: Position, q: Position):
        assert isinstance(state.grid[p], MovingObstacle)
        moved_obstacles.append(state.grid[p])

        # perform real swap
        Grid.swap(state.grid, p, q)

    with patch.object(state.grid, 'swap', swap_patch):
        move_obstacles(state, Action.PICK_N_DROP)

    # four unique and distinct moving-objects
    moved_obstacle_ids = [id(obj) for obj in moved_obstacles]
    assert len(moved_obstacle_ids) == len(set(moved_obstacle_ids)) == 4


@pytest.mark.parametrize(
    'objects,expected_objects',
    [
        (
            [[Floor(), MovingObstacle(), MovingObstacle()]],
            [[MovingObstacle(), MovingObstacle(), Floor()]],
        ),
        (
            [[MovingObstacle(), Floor(), MovingObstacle()]],
            [[Floor(), MovingObstacle(), MovingObstacle()]],
        ),
        (
            [[MovingObstacle(), MovingObstacle(), Floor()]],
            [[MovingObstacle(), Floor(), MovingObstacle()]],
        ),
    ],
)
def test_move_obstacles(
    objects: Sequence[Sequence[GridObject]],
    expected_objects: Sequence[Sequence[GridObject]],
):
    state = State(Grid(objects), MagicMock())
    expected_state = State(Grid(expected_objects), MagicMock())

    action = MagicMock()
    move_obstacles(state, action)
    assert state.grid == expected_state.grid


@pytest.mark.parametrize(
    'door_state,door_color,key_color,action,expected_state',
    [
        # LOCKED
        (
            Door.Status.LOCKED,
            Color.RED,
            Color.RED,
            Action.ACTUATE,
            Door.Status.OPEN,
        ),
        (
            Door.Status.LOCKED,
            Color.RED,
            Color.BLUE,
            Action.ACTUATE,
            Door.Status.LOCKED,
        ),
        # CLOSED
        (
            Door.Status.CLOSED,
            Color.RED,
            Color.BLUE,
            Action.ACTUATE,
            Door.Status.OPEN,
        ),
        # OPEN
        (
            Door.Status.OPEN,
            Color.RED,
            Color.RED,
            Action.ACTUATE,
            Door.Status.OPEN,
        ),
        # not ACTUATE
        (
            Door.Status.LOCKED,
            Color.RED,
            Color.RED,
            Action.PICK_N_DROP,
            Door.Status.LOCKED,
        ),
        (
            Door.Status.LOCKED,
            Color.RED,
            Color.BLUE,
            Action.PICK_N_DROP,
            Door.Status.LOCKED,
        ),
        (
            Door.Status.CLOSED,
            Color.RED,
            Color.BLUE,
            Action.PICK_N_DROP,
            Door.Status.CLOSED,
        ),
        (
            Door.Status.OPEN,
            Color.RED,
            Color.RED,
            Action.PICK_N_DROP,
            Door.Status.OPEN,
        ),
    ],
)
def test_actuate_door(
    door_state: Door.Status,
    door_color: Color,
    key_color: Color,
    action: Action,
    expected_state: Door.Status,
):
    # agent facing door
    grid = Grid.from_shape((2, 1))
    grid[0, 0] = door = Door(door_state, door_color)
    agent = Agent(Position(1, 0), Orientation.N, Key(key_color))
    state = State(grid, agent)

    actuate_door(state, action)
    assert door.state == expected_state

    # agent facing away
    grid = Grid.from_shape((2, 1))
    grid[0, 0] = door = Door(door_state, door_color)
    agent = Agent(Position(1, 0), Orientation.S, Key(key_color))
    state = State(grid, agent)

    actuate_door(state, action)
    assert door.state == door_state


@pytest.mark.parametrize(
    'content,orientation,action,expected',
    [
        # empty box
        (Floor(), Orientation.N, Action.ACTUATE, True),
        (Floor(), Orientation.S, Action.ACTUATE, False),
        (Floor(), Orientation.N, Action.PICK_N_DROP, False),
        (Floor(), Orientation.S, Action.PICK_N_DROP, False),
        # content is key
        (Key(Color.RED), Orientation.N, Action.ACTUATE, True),
        (Key(Color.RED), Orientation.S, Action.ACTUATE, False),
        (Key(Color.RED), Orientation.N, Action.PICK_N_DROP, False),
        (Key(Color.RED), Orientation.S, Action.PICK_N_DROP, False),
    ],
)
def test_actuate_box(
    content: GridObject,
    orientation: Orientation,
    action: Action,
    expected: bool,
):
    grid = Grid.from_shape((2, 1))
    grid[0, 0] = box = Box(content)
    agent = Agent(Position(1, 0), orientation)
    state = State(grid, agent)

    actuate_box(state, action)
    assert (grid[0, 0] is box) != expected
    assert (grid[0, 0] is content) == expected


@pytest.mark.parametrize(
    'orientation,action',
    [
        # empty box
        (Orientation.N, Action.ACTUATE),
        (Orientation.S, Action.ACTUATE),
        (Orientation.N, Action.PICK_N_DROP),
        (Orientation.S, Action.PICK_N_DROP),
        # content is key
        (Orientation.N, Action.ACTUATE),
        (Orientation.S, Action.ACTUATE),
        (Orientation.N, Action.PICK_N_DROP),
        (Orientation.S, Action.PICK_N_DROP),
    ],
)
def test_actuate_no_box(
    orientation: Orientation,
    action: Action,
):
    grid = Grid.from_shape((2, 1))
    agent = Agent(Position(1, 0), orientation)
    state = State(grid, agent)

    prev_state = copy.deepcopy(state)
    actuate_box(state, action)
    assert prev_state == state


@pytest.mark.parametrize(
    'position_telepod1,position_telepod2,position_agent,expected',
    [
        (Position(0, 0), Position(1, 1), Position(0, 0), Position(1, 1)),
        (Position(0, 0), Position(1, 1), Position(0, 1), Position(0, 1)),
        (Position(0, 0), Position(1, 1), Position(1, 0), Position(1, 0)),
        (Position(0, 0), Position(1, 1), Position(1, 1), Position(0, 0)),
        (Position(1, 1), Position(0, 0), Position(0, 0), Position(1, 1)),
        (Position(1, 1), Position(0, 0), Position(0, 1), Position(0, 1)),
        (Position(1, 1), Position(0, 0), Position(1, 0), Position(1, 0)),
        (Position(1, 1), Position(0, 0), Position(1, 1), Position(0, 0)),
    ],
)
def test_teleport(
    position_telepod1: Position,
    position_telepod2: Position,
    position_agent: Position,
    expected: Position,
):
    grid = Grid.from_shape((2, 2))
    grid[position_telepod1] = Telepod(Color.RED)
    grid[position_telepod2] = Telepod(Color.RED)

    agent = Agent(position_agent, Orientation.N)
    state = State(grid, agent)

    teleport(state, Action.ACTUATE)
    assert state.agent.position == expected


@pytest.mark.parametrize(
    'name,kwargs',
    [
        ('chain', {'transition_functions': []}),
        ('pickndrop', {}),
        ('move_obstacles', {}),
        ('actuate_door', {}),
        ('actuate_box', {}),
        ('teleport', {}),
    ],
)
def test_factory_valid(name: str, kwargs):
    factory(name, **kwargs)


@pytest.mark.parametrize(
    'name,kwargs',
    [
        ('chain', {}),
        ('invalid', {}),
    ],
)
def test_factory_invalid(name: str, kwargs):
    with pytest.raises(ValueError):
        factory(name, **kwargs)
