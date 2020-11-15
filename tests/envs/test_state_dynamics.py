""" Tests state dynamics """

import copy
import random
from typing import Sequence

import pytest

from gym_gridverse.actions import Actions
from gym_gridverse.envs.reset_functions import reset_minigrid_dynamic_obstacles
from gym_gridverse.envs.state_dynamics import (
    factory,
    move_agent,
    pickup_mechanics,
    rotate_agent,
    step_objects,
)
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Floor,
    Key,
    MovingObstacle,
    NoneGridObject,
    Wall,
)
from gym_gridverse.info import Agent, Grid, Orientation, Position
from gym_gridverse.state import State


@pytest.mark.parametrize(
    'agent,actions,expected',
    [
        # Facing north, rotate to WEST
        (
            Agent(
                Position(random.randint(0, 10), random.randint(0, 10)),
                Orientation.N,
                NoneGridObject(),
            ),
            [Actions.TURN_LEFT],
            Orientation.W,
        ),
        # Not rotating
        (
            Agent(
                Position(random.randint(0, 10), random.randint(0, 10)),
                Orientation.W,
                NoneGridObject(),
            ),
            [Actions.MOVE_LEFT, Actions.ACTUATE, Actions.PICK_N_DROP],
            Orientation.W,
        ),
        # Two rotation to EAST
        (
            Agent(
                Position(random.randint(0, 10), random.randint(0, 10)),
                Orientation.W,
                NoneGridObject(),
            ),
            [Actions.TURN_LEFT, Actions.TURN_LEFT],
            Orientation.E,
        ),
        # One back to SOUTH
        (
            Agent(
                Position(random.randint(0, 10), random.randint(0, 10)),
                Orientation.E,
                NoneGridObject(),
            ),
            [Actions.TURN_RIGHT],
            Orientation.S,
        ),
        # Full circle for fun to SOUTH
        (
            Agent(
                Position(random.randint(0, 10), random.randint(0, 10)),
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


# TODO clean fixture
@pytest.fixture
def grid() -> Grid:
    """Sets up a 3x2 Grid"""
    return Grid(height=3, width=2)


# TODO clean fixture
@pytest.fixture
def agent() -> Agent:
    """Sets up an agent facing north in (2,1)"""
    return Agent(position=Position(2, 1), orientation=Orientation.N)


# TODO parametrize
def test_move_action_basic_movement_north(
    grid: Grid, agent: Agent,  # pylint: disable=redefined-outer-name
):
    """Test unblocked movement"""

    # Move to (2,0)
    move_agent(agent, grid, action=Actions.MOVE_LEFT)
    assert agent.position == Position(2, 0)

    # Move to (2,1)
    move_agent(agent, grid, action=Actions.MOVE_RIGHT)
    assert agent.position == Position(2, 1)

    # Move up twice to (0,1)
    move_agent(agent, grid, action=Actions.MOVE_FORWARD)
    move_agent(agent, grid, action=Actions.MOVE_FORWARD)
    assert agent.position == Position(0, 1)

    # Move down once to (1,1)
    move_agent(agent, grid, action=Actions.MOVE_BACKWARD)
    assert agent.position == Position(1, 1)


# TODO parametrize
def test_move_action_blocked_by_grid_object(
    grid: Grid, agent: Agent,  # pylint: disable=redefined-outer-name
):
    """ Puts an object on (2,0) and try to move there"""

    grid[Position(2, 0)] = Door(Door.Status.CLOSED, Colors.YELLOW)
    move_agent(agent, grid, action=Actions.MOVE_LEFT)

    assert agent.position == Position(2, 1)


# TODO parametrize
def test_move_action_blocked_by_edges(
    grid: Grid, agent: Agent,  # pylint: disable=redefined-outer-name
):
    """Verify agent does not move outside of bounds"""

    move_agent(agent, grid, action=Actions.MOVE_RIGHT)

    assert agent.position == Position(2, 1)


# TODO parametrize
def test_move_action_other_actions_are_inactive(
    grid: Grid, agent: Agent,  # pylint: disable=redefined-outer-name
):
    """Make sure actions that do not _move_ are ignored"""

    move_agent(agent, grid, action=Actions.TURN_RIGHT)
    move_agent(agent, grid, action=Actions.PICK_N_DROP)
    move_agent(agent, grid, action=Actions.ACTUATE)

    assert agent.position == Position(2, 1)


# TODO parametrize
def test_move_action_facing_east(
    grid: Grid, agent: Agent,  # pylint: disable=redefined-outer-name
):
    """Just another test to make sure orientation works as intended"""
    agent.orientation = Orientation.E

    move_agent(agent, grid, action=Actions.MOVE_LEFT)
    move_agent(agent, grid, action=Actions.MOVE_LEFT)
    assert agent.position == Position(0, 1)

    move_agent(agent, grid, action=Actions.MOVE_BACKWARD)
    assert agent.position == Position(0, 0)

    move_agent(agent, grid, action=Actions.MOVE_FORWARD)
    assert agent.position == Position(0, 1)
    move_agent(agent, grid, action=Actions.MOVE_FORWARD)
    assert agent.position == Position(0, 1)

    move_agent(agent, grid, action=Actions.MOVE_RIGHT)
    assert agent.position == Position(1, 1)


# TODO parametrize
def test_move_action_can_go_on_non_block_objects(
    grid: Grid, agent: Agent,  # pylint: disable=redefined-outer-name
):
    grid[Position(2, 0)] = Door(Door.Status.OPEN, Colors.YELLOW)
    move_agent(agent, grid, action=Actions.MOVE_LEFT)
    assert agent.position == Position(2, 0)

    grid[Position(2, 1)] = Key(Colors.BLUE)
    move_agent(agent, grid, action=Actions.MOVE_RIGHT)
    assert agent.position == Position(2, 1)


def step_with_copy(state: State, action: Actions) -> State:
    next_state = copy.deepcopy(state)
    pickup_mechanics(next_state, action)
    return next_state


def test_pickup_mechanics_nothing_to_pickup():
    grid = Grid(height=3, width=4)
    agent = Agent(position=Position(1, 2), orientation=Orientation.S)
    item_pos = Position(2, 2)

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
    agent = Agent(position=Position(1, 2), orientation=Orientation.S)
    item_pos = Position(2, 2)

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
    agent = Agent(position=Position(1, 2), orientation=Orientation.S)
    item_pos = Position(2, 2)

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
    agent = Agent(position=Position(1, 2), orientation=Orientation.S)
    item_pos = Position(2, 2)

    agent.obj = Key(Colors.BLUE)
    grid[item_pos] = Key(Colors.GREEN)
    state = State(grid, agent)

    next_state = step_with_copy(state, Actions.PICK_N_DROP)
    assert state.grid[item_pos] == next_state.agent.obj
    assert state.agent.obj == next_state.grid[item_pos]


def test_step_called_once():
    """Tests step is called exactly once on all objects

    There was this naive implementation that looped over all positions, and
    called `.step()` on it. Unfortunately, when `step` caused the object to
    _move_, then there was a possibility that the object moved to a later
    position, hence being called twice. This test is here to make sure that
    does not happen again.
    """

    def call_counter(func):
        def helper(*args, **kwargs):
            helper.count += 1
            return func(*args, **kwargs)

        helper.count = 0
        return helper

    w, h, n = 6, 6, 4
    state = reset_minigrid_dynamic_obstacles(h, w, n)

    # replace obstacles with those that count step
    obstacles = []
    for pos in state.grid.positions():
        if isinstance(state.grid[pos], MovingObstacle):
            state.grid[pos].step = call_counter(state.grid[pos].step)
            obstacles.append(state.grid[pos])

    step_objects(state, Actions.PICK_N_DROP)

    for obs in obstacles:
        assert obs.step.count == 1


@pytest.mark.parametrize(
    'name',
    ['update_agent', 'step_objects', 'actuate_mechanics', 'pickup_mechanics'],
)
def test_factory_valid(name):
    factory(name)


def test_factory_invalid():
    with pytest.raises(ValueError):
        factory('invalid')
