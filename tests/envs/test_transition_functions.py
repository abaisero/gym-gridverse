""" Tests state dynamics """

import copy
import random
from typing import Optional, Sequence
from unittest.mock import MagicMock, patch

import numpy.random as rnd
import pytest

from gym_gridverse.actions import Actions
from gym_gridverse.envs.reset_functions import reset_minigrid_dynamic_obstacles
from gym_gridverse.envs.transition_functions import (
    _step_moving_obstacle,
    actuate_box,
    actuate_door,
    factory,
    move_agent,
    pickup_mechanics,
    rotate_agent,
    step_moving_obstacles,
)
from gym_gridverse.geometry import Position
from gym_gridverse.grid_object import (
    Box,
    Colors,
    Door,
    Floor,
    GridObject,
    Key,
    MovingObstacle,
    NoneGridObject,
    Wall,
)
from gym_gridverse.info import Agent, Grid, Orientation, PositionOrTuple
from gym_gridverse.state import State


def make_moving_obstacle_state():
    grid = Grid(3, 3)
    grid[1, 1] = MovingObstacle()
    agent = MagicMock()
    return State(grid, agent)


@pytest.mark.parametrize(
    'agent,actions,expected',
    [
        # Facing north, rotate to WEST
        (
            Agent(
                (random.randint(0, 10), random.randint(0, 10)),
                Orientation.N,
                NoneGridObject(),
            ),
            [Actions.TURN_LEFT],
            Orientation.W,
        ),
        # Not rotating
        (
            Agent(
                (random.randint(0, 10), random.randint(0, 10)),
                Orientation.W,
                NoneGridObject(),
            ),
            [Actions.MOVE_LEFT, Actions.ACTUATE, Actions.PICK_N_DROP],
            Orientation.W,
        ),
        # Two rotation to EAST
        (
            Agent(
                (random.randint(0, 10), random.randint(0, 10)),
                Orientation.W,
                NoneGridObject(),
            ),
            [Actions.TURN_LEFT, Actions.TURN_LEFT],
            Orientation.E,
        ),
        # One back to SOUTH
        (
            Agent(
                (random.randint(0, 10), random.randint(0, 10)),
                Orientation.E,
                NoneGridObject(),
            ),
            [Actions.TURN_RIGHT],
            Orientation.S,
        ),
        # Full circle for fun to SOUTH
        (
            Agent(
                (random.randint(0, 10), random.randint(0, 10)),
                Orientation.S,
                NoneGridObject(),
            ),
            [
                Actions.TURN_RIGHT,
                Actions.TURN_RIGHT,
                Actions.TURN_RIGHT,
                Actions.TURN_RIGHT,
            ],
            Orientation.S,
        ),
    ],
)
def test_rotate_agent(
    agent: Agent,  # pylint: disable=redefined-outer-name
    actions: Sequence[Actions],
    expected: Orientation,
):
    for action in actions:
        rotate_agent(agent, action)

    assert agent.orientation == expected


@pytest.mark.parametrize(
    'position,orientation,actions,expected',
    [
        #  unblocked movement
        ((2, 1), Orientation.N, [Actions.MOVE_LEFT], (2, 0)),
        ((2, 0), Orientation.N, [Actions.MOVE_RIGHT], (2, 1)),
        (
            (2, 1),
            Orientation.N,
            [Actions.MOVE_FORWARD, Actions.MOVE_FORWARD],
            (0, 1),
        ),
        ((0, 1), Orientation.N, [Actions.MOVE_BACKWARD], (1, 1)),
        # blocked by grid object
        # blocked by edges
        ((2, 1), Orientation.N, [Actions.MOVE_RIGHT], (2, 1)),
        # non-movements
        ((2, 1), Orientation.N, [Actions.TURN_RIGHT], (2, 1)),
        ((2, 1), Orientation.N, [Actions.TURN_LEFT], (2, 1)),
        ((2, 1), Orientation.N, [Actions.ACTUATE], (2, 1)),
        ((2, 1), Orientation.N, [Actions.PICK_N_DROP], (2, 1)),
        # facing east
        ((2, 1), Orientation.E, [Actions.MOVE_LEFT, Actions.MOVE_LEFT], (0, 1)),
        ((0, 1), Orientation.E, [Actions.MOVE_BACKWARD], (0, 0)),
        ((0, 0), Orientation.E, [Actions.MOVE_FORWARD], (0, 1)),
        ((0, 1), Orientation.E, [Actions.MOVE_FORWARD], (0, 1)),
        ((0, 1), Orientation.E, [Actions.MOVE_RIGHT], (1, 1)),
    ],
)
def test_move_action(
    position: PositionOrTuple,
    orientation: Orientation,
    actions: Sequence[Actions],
    expected: PositionOrTuple,
):
    grid = Grid(height=3, width=2)
    agent = Agent(position=position, orientation=orientation)

    for action in actions:
        move_agent(agent, grid, action=action)

    assert agent.position == expected


# TODO integrate with previous test
def test_move_action_blocked_by_grid_object():
    """ Puts an object on (2,0) and try to move there"""
    grid = Grid(height=3, width=2)
    agent = Agent(position=(2, 1), orientation=Orientation.N)

    grid[2, 0] = Door(Door.Status.CLOSED, Colors.YELLOW)
    move_agent(agent, grid, action=Actions.MOVE_LEFT)

    assert agent.position == (2, 1)


# TODO integrate with previous test
def test_move_action_can_go_on_non_block_objects():
    grid = Grid(height=3, width=2)
    agent = Agent(position=(2, 1), orientation=Orientation.N)

    grid[2, 0] = Door(Door.Status.OPEN, Colors.YELLOW)
    move_agent(agent, grid, action=Actions.MOVE_LEFT)
    assert agent.position == (2, 0)

    grid[2, 1] = Key(Colors.BLUE)
    move_agent(agent, grid, action=Actions.MOVE_RIGHT)
    assert agent.position == (2, 1)


def step_with_copy(state: State, action: Actions) -> State:
    next_state = copy.deepcopy(state)
    pickup_mechanics(next_state, action)
    return next_state


def test_pickup_mechanics_nothing_to_pickup():
    grid = Grid(height=3, width=4)
    agent = Agent(position=(1, 2), orientation=Orientation.S)
    item_pos = (2, 2)

    state = State(grid, agent)

    # Cannot pickup floor
    next_state = step_with_copy(state, Actions.PICK_N_DROP)
    assert state == next_state

    # Cannot pickup door
    grid[item_pos] = Door(Door.Status.CLOSED, Colors.GREEN)
    next_state = step_with_copy(state, Actions.PICK_N_DROP)
    assert state == next_state

    assert isinstance(next_state.grid[item_pos], Door)


def test_pickup_mechanics_pickup():
    grid = Grid(height=3, width=4)
    agent = Agent(position=(1, 2), orientation=Orientation.S)
    item_pos = (2, 2)

    grid[item_pos] = Key(Colors.GREEN)
    state = State(grid, agent)

    # Pick up works
    next_state = step_with_copy(state, Actions.PICK_N_DROP)
    assert grid[item_pos] == next_state.agent.obj
    assert isinstance(next_state.grid[item_pos], Floor)

    # Pick up only works with correct action
    next_state = step_with_copy(state, Actions.MOVE_LEFT)
    assert grid[item_pos] != next_state.agent.obj
    assert next_state.grid[item_pos] == grid[item_pos]


def test_pickup_mechanics_drop():
    grid = Grid(height=3, width=4)
    agent = Agent(position=(1, 2), orientation=Orientation.S)
    item_pos = (2, 2)

    agent.obj = Key(Colors.BLUE)
    state = State(grid, agent)

    # Can drop:
    next_state = step_with_copy(state, Actions.PICK_N_DROP)
    assert isinstance(next_state.agent.obj, NoneGridObject)
    assert agent.obj == next_state.grid[item_pos]

    # Cannot drop:
    state.grid[item_pos] = Wall()

    next_state = step_with_copy(state, Actions.PICK_N_DROP)
    assert isinstance(next_state.grid[item_pos], Wall)
    assert agent.obj == next_state.agent.obj


def test_pickup_mechanics_swap():
    grid = Grid(height=3, width=4)
    agent = Agent(position=(1, 2), orientation=Orientation.S)
    item_pos = (2, 2)

    agent.obj = Key(Colors.BLUE)
    grid[item_pos] = Key(Colors.GREEN)
    state = State(grid, agent)

    next_state = step_with_copy(state, Actions.PICK_N_DROP)
    assert state.grid[item_pos] == next_state.agent.obj
    assert state.agent.obj == next_state.grid[item_pos]


def test_step_moving_obstacles_once_per_obstacle():
    """Tests step is called exactly once on all moving obstacles

    There was this naive implementation that looped over all positions, and
    called `.step()` on it. Unfortunately, when `step` caused the object to
    _move_, then there was a possibility that the object moved to a later
    position, hence being called twice. This test is here to make sure that
    does not happen again.
    """
    state = reset_minigrid_dynamic_obstacles(height=6, width=6, num_obstacles=4)

    counts = {}

    def patched_step_moving_obstacle(
        grid: Grid, position: Position, *, rng: Optional[rnd.Generator] = None
    ):
        key = id(grid[position])
        try:
            counts[key] += 1
        except KeyError:
            counts[key] = 1

        _step_moving_obstacle(grid, position, rng=rng)

    with patch(
        'gym_gridverse.envs.transition_functions._step_moving_obstacle',
        new=patched_step_moving_obstacle,
    ):
        step_moving_obstacles(state, Actions.PICK_N_DROP)

    assert len(counts) == 4
    assert all(count == 1 for count in counts.values())


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
def test_step_moving_obstacles(
    objects: Sequence[Sequence[GridObject]],
    expected_objects: Sequence[Sequence[GridObject]],
):
    state = State(Grid.from_objects(objects), MagicMock())
    expected_state = State(Grid.from_objects(expected_objects), MagicMock())

    action = MagicMock()
    step_moving_obstacles(state, action)
    assert state.grid == expected_state.grid


@pytest.mark.parametrize(
    'door_state,door_color,key_color,action,expected_state',
    [
        # LOCKED
        (
            Door.Status.LOCKED,
            Colors.RED,
            Colors.RED,
            Actions.ACTUATE,
            Door.Status.OPEN,
        ),
        (
            Door.Status.LOCKED,
            Colors.RED,
            Colors.BLUE,
            Actions.ACTUATE,
            Door.Status.LOCKED,
        ),
        # CLOSED
        (
            Door.Status.CLOSED,
            Colors.RED,
            Colors.BLUE,
            Actions.ACTUATE,
            Door.Status.OPEN,
        ),
        # OPEN
        (
            Door.Status.OPEN,
            Colors.RED,
            Colors.RED,
            Actions.ACTUATE,
            Door.Status.OPEN,
        ),
        # not ACTUATE
        (
            Door.Status.LOCKED,
            Colors.RED,
            Colors.RED,
            Actions.PICK_N_DROP,
            Door.Status.LOCKED,
        ),
        (
            Door.Status.LOCKED,
            Colors.RED,
            Colors.BLUE,
            Actions.PICK_N_DROP,
            Door.Status.LOCKED,
        ),
        (
            Door.Status.CLOSED,
            Colors.RED,
            Colors.BLUE,
            Actions.PICK_N_DROP,
            Door.Status.CLOSED,
        ),
        (
            Door.Status.OPEN,
            Colors.RED,
            Colors.RED,
            Actions.PICK_N_DROP,
            Door.Status.OPEN,
        ),
    ],
)
def test_actuate_door(
    door_state: Door.Status,
    door_color: Colors,
    key_color: Colors,
    action: Actions,
    expected_state: Door.Status,
):
    # agent facing door
    grid = Grid(2, 1)
    grid[0, 0] = door = Door(door_state, door_color)
    agent = Agent((1, 0), Orientation.N, Key(key_color))
    state = State(grid, agent)

    actuate_door(state, action)
    assert door.state == expected_state

    # agent facing away
    grid = Grid(2, 1)
    grid[0, 0] = door = Door(door_state, door_color)
    agent = Agent((1, 0), Orientation.S, Key(key_color))
    state = State(grid, agent)

    actuate_door(state, action)
    assert door.state == door_state


@pytest.mark.parametrize(
    'content,orientation,action,expected',
    [
        # empty box
        (Floor(), Orientation.N, Actions.ACTUATE, True),
        (Floor(), Orientation.S, Actions.ACTUATE, False),
        (Floor(), Orientation.N, Actions.PICK_N_DROP, False),
        (Floor(), Orientation.S, Actions.PICK_N_DROP, False),
        # content is key
        (Key(Colors.RED), Orientation.N, Actions.ACTUATE, True),
        (Key(Colors.RED), Orientation.S, Actions.ACTUATE, False),
        (Key(Colors.RED), Orientation.N, Actions.PICK_N_DROP, False),
        (Key(Colors.RED), Orientation.S, Actions.PICK_N_DROP, False),
    ],
)
def test_actuate_box(
    content: GridObject,
    orientation: Orientation,
    action: Actions,
    expected: bool,
):
    grid = Grid(2, 1)
    grid[0, 0] = box = Box(content)
    agent = Agent((1, 0), orientation)
    state = State(grid, agent)

    actuate_box(state, action)
    assert (grid[0, 0] is box) != expected
    assert (grid[0, 0] is content) == expected


@pytest.mark.parametrize(
    'name,kwargs',
    [
        ('chain', {'transition_functions': []}),
        ('update_agent', {}),
        ('pickup_mechanics', {}),
        ('step_moving_obstacles', {}),
        ('actuate_door', {}),
        ('actuate_box', {}),
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
