from typing import Optional, Type

import pytest

from gym_gridverse.action import Action
from gym_gridverse.agent import Agent
from gym_gridverse.envs.reward_functions import (
    actuate_door,
    bump_into_wall,
    bump_moving_obstacle,
    factory,
    getting_closer,
    getting_closer_shortest_path,
    living_reward,
    pickndrop,
    proportional_to_distance,
    reach_exit,
)
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import (
    Color,
    Door,
    Exit,
    Key,
    MovingObstacle,
    Wall,
)
from gym_gridverse.state import State


def make_5x5_exit_state() -> State:
    """makes a simple 5x5 state with exit object in the middle"""
    grid = Grid.from_shape((5, 5))
    grid[2, 2] = Exit()
    agent = Agent(Position(0, 0), Orientation.F)
    return State(grid, agent)


def make_exit_state(agent_on_exit: bool) -> State:
    """makes a simple state with a wall in front of the agent"""
    grid = Grid.from_shape((2, 1))
    grid[0, 0] = Exit()
    agent_position = Position(0, 0) if agent_on_exit else Position(1, 0)
    agent = Agent(agent_position, Orientation.F)
    return State(grid, agent)


def make_wall_state(orientation: Orientation = Orientation.F) -> State:
    """makes a simple state with exit object and agent on or off the exit"""
    grid = Grid.from_shape((2, 1))
    grid[0, 0] = Wall()
    agent = Agent(Position(1, 0), orientation)
    return State(grid, agent)


def make_door_state(door_status: Optional[Door.Status]) -> State:
    """makes a simple state with a door"""
    grid = Grid.from_shape((2, 1))

    if door_status:
        grid[0, 0] = Door(door_status, Color.RED)

    agent = Agent(Position(1, 0), Orientation.F)
    return State(grid, agent)


def make_key_state(has_key: bool) -> State:
    """makes a simple state with a door"""
    grid = Grid.from_shape((1, 1))
    obj = Key(Color.RED) if has_key else None
    agent = Agent(Position(0, 0), Orientation.F, obj)
    return State(grid, agent)


def make_moving_obstacle_state(agent_on_obstacle: bool) -> State:
    """makes a simple state with exit object and agent on or off the exit"""
    grid = Grid.from_shape((2, 1))
    grid[0, 0] = MovingObstacle()
    agent_position = Position(0, 0) if agent_on_obstacle else Position(1, 0)
    agent = Agent(agent_position, Orientation.F)
    return State(grid, agent)


@pytest.mark.parametrize(
    'kwargs,expected',
    [
        ({}, -1.0),
        ({'reward': -1.0}, -1.0),
        ({'reward': 0.0}, 0.0),
        ({'reward': 1.0}, 1.0),
    ],
)
def test_living_reward(
    kwargs, expected: float, forbidden_state_maker, forbidden_action_maker
):
    state = forbidden_state_maker()
    action = forbidden_action_maker()
    next_state = forbidden_state_maker()
    assert living_reward(state, action, next_state, **kwargs) == expected


@pytest.mark.parametrize(
    'agent_on_exit,kwargs,expected',
    [
        (True, {}, 1.0),
        (False, {}, 0.0),
        (True, {'reward_on': 10.0, 'reward_off': -1.0}, 10.0),
        (False, {'reward_on': 10.0, 'reward_off': -1.0}, -1.0),
    ],
)
def test_reach_exit(
    agent_on_exit: bool,
    kwargs,
    expected: float,
    forbidden_state_maker,
    forbidden_action_maker,
):
    state = forbidden_state_maker()
    action = forbidden_action_maker()
    next_state = make_exit_state(agent_on_exit=agent_on_exit)
    assert reach_exit(state, action, next_state, **kwargs) == expected


@pytest.mark.parametrize(
    'agent_on_obstacle,kwargs,expected',
    [
        (True, {}, -1.0),
        (False, {}, 0.0),
        (True, {'reward': -10.0}, -10.0),
        (False, {'reward': -10.0}, 0.0),
    ],
)
def test_bump_moving_obstacle_default(
    agent_on_obstacle: bool,
    kwargs,
    expected: float,
    forbidden_state_maker,
    forbidden_action_maker,
):
    state = forbidden_state_maker()
    action = forbidden_action_maker()
    next_state = make_moving_obstacle_state(agent_on_obstacle)
    assert bump_moving_obstacle(state, action, next_state, **kwargs) == expected


@pytest.mark.parametrize(
    'position,kwargs,expected',
    [
        # moving agent on the top row
        (Position(0, 0), {}, -4.0),
        (Position(0, 1), {}, -3.0),
        (Position(0, 2), {}, -2.0),
        (Position(0, 3), {}, -3.0),
        (Position(0, 4), {}, -4.0),
        # moving agent on the middle row
        (Position(2, 0), {}, -2.0),
        (Position(2, 1), {}, -1.0),
        (Position(2, 2), {}, 0.0),
        (Position(2, 3), {}, -1.0),
        (Position(2, 4), {}, -2.0),
        # moving agent on the top row
        (Position(0, 0), {'reward_per_unit_distance': 0.1}, 0.40),
        (Position(0, 1), {'reward_per_unit_distance': 0.1}, 0.30),
        (Position(0, 2), {'reward_per_unit_distance': 0.1}, 0.20),
        (Position(0, 3), {'reward_per_unit_distance': 0.1}, 0.30),
        (Position(0, 4), {'reward_per_unit_distance': 0.1}, 0.40),
        # moving agent on the middle row
        (Position(2, 0), {'reward_per_unit_distance': 0.1}, 0.20),
        (Position(2, 1), {'reward_per_unit_distance': 0.1}, 0.10),
        (Position(2, 2), {'reward_per_unit_distance': 0.1}, 0.0),
        (Position(2, 3), {'reward_per_unit_distance': 0.1}, 0.10),
        (Position(2, 4), {'reward_per_unit_distance': 0.1}, 0.20),
    ],
)
def test_proportional_to_distance_default(
    position: Position,
    kwargs,
    expected: float,
    forbidden_state_maker,
    forbidden_action_maker,
):
    state = forbidden_state_maker()
    action = forbidden_action_maker()
    next_state = make_5x5_exit_state()
    # TODO: find better way to construct this state
    next_state.agent.position = position

    reward = proportional_to_distance(
        state, action, next_state, object_type=Exit, **kwargs
    )
    assert round(reward, 7) == expected


@pytest.mark.parametrize(
    'agent_on_exit,next_agent_on_exit,kwargs,expected',
    [
        (False, False, {}, 0.0),
        (False, True, {}, 1.0),
        (True, False, {}, -1.0),
        (True, True, {}, 0.0),
        (False, False, {'reward_closer': 2.0, 'reward_further': -5.0}, 0.0),
        (False, True, {'reward_closer': 2.0, 'reward_further': -5.0}, 2.0),
        (True, False, {'reward_closer': 2.0, 'reward_further': -5.0}, -5.0),
        (True, True, {'reward_closer': 2.0, 'reward_further': -5.0}, 0.0),
    ],
)
def test_getting_closer(
    agent_on_exit: bool,
    next_agent_on_exit: bool,
    kwargs,
    expected: float,
    forbidden_action_maker,
):
    state = make_exit_state(agent_on_exit)
    action = forbidden_action_maker()
    next_state = make_exit_state(next_agent_on_exit)
    assert (
        getting_closer(state, action, next_state, object_type=Exit, **kwargs)
        == expected
    )


@pytest.mark.parametrize(
    'agent_on_exit,next_agent_on_exit,kwargs,expected',
    [
        (False, False, {}, 0.0),
        (False, True, {}, 1.0),
        (True, False, {}, -1.0),
        (True, True, {}, 0.0),
        (False, False, {'reward_closer': 2.0, 'reward_further': -5.0}, 0.0),
        (False, True, {'reward_closer': 2.0, 'reward_further': -5.0}, 2.0),
        (True, False, {'reward_closer': 2.0, 'reward_further': -5.0}, -5.0),
        (True, True, {'reward_closer': 2.0, 'reward_further': -5.0}, 0.0),
    ],
)
def test_getting_closer_shortest_path(
    agent_on_exit: bool,
    next_agent_on_exit: bool,
    kwargs,
    expected: float,
    forbidden_action_maker,
):
    state = make_exit_state(agent_on_exit)
    action = forbidden_action_maker()
    next_state = make_exit_state(next_agent_on_exit)
    assert (
        getting_closer_shortest_path(
            state, action, next_state, object_type=Exit, **kwargs
        )
        == expected
    )


@pytest.mark.parametrize(
    'state,action,kwargs,expected',
    [
        # not bumping
        (make_wall_state(), Action.MOVE_LEFT, {}, 0.0),
        (make_wall_state(), Action.PICK_N_DROP, {}, 0.0),
        # bumping
        (make_wall_state(Orientation.F), Action.MOVE_FORWARD, {}, -1.0),
        (make_wall_state(Orientation.R), Action.MOVE_LEFT, {}, -1.0),
        (make_wall_state(Orientation.B), Action.MOVE_BACKWARD, {}, -1.0),
        (make_wall_state(Orientation.L), Action.MOVE_RIGHT, {}, -1.0),
        # reward value
        (make_wall_state(), Action.MOVE_FORWARD, {'reward': -4.78}, -4.78),
    ],
)
def test_bump_into_wall(
    state: State,
    action: Action,
    kwargs,
    expected: float,
    forbidden_state_maker,
):
    next_state = forbidden_state_maker()
    assert bump_into_wall(state, action, next_state, **kwargs) == expected


@pytest.mark.parametrize(
    'door_status,next_door_status,action,kwargs,expected',
    [
        # not opening
        (Door.Status.CLOSED, Door.Status.OPEN, Action.MOVE_LEFT, {}, 0.0),
        (Door.Status.CLOSED, Door.Status.OPEN, Action.PICK_N_DROP, {}, 0.0),
        # default values
        (Door.Status.OPEN, Door.Status.OPEN, Action.ACTUATE, {}, 0.0),
        (Door.Status.OPEN, Door.Status.CLOSED, Action.ACTUATE, {}, -1.0),
        (Door.Status.OPEN, Door.Status.LOCKED, Action.ACTUATE, {}, -1.0),
        (Door.Status.CLOSED, Door.Status.OPEN, Action.ACTUATE, {}, 1.0),
        (Door.Status.CLOSED, Door.Status.CLOSED, Action.ACTUATE, {}, 0.0),
        (Door.Status.CLOSED, Door.Status.LOCKED, Action.ACTUATE, {}, 0.0),
        (Door.Status.LOCKED, Door.Status.OPEN, Action.ACTUATE, {}, 1.0),
        (Door.Status.LOCKED, Door.Status.CLOSED, Action.ACTUATE, {}, 0.0),
        (Door.Status.LOCKED, Door.Status.LOCKED, Action.ACTUATE, {}, 0.0),
        # custom values
        (
            Door.Status.OPEN,
            Door.Status.OPEN,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            0.0,
        ),
        (
            Door.Status.OPEN,
            Door.Status.CLOSED,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            1.5,
        ),
        (
            Door.Status.OPEN,
            Door.Status.LOCKED,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            1.5,
        ),
        (
            Door.Status.CLOSED,
            Door.Status.OPEN,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            -1.5,
        ),
        (
            Door.Status.CLOSED,
            Door.Status.CLOSED,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            0.0,
        ),
        (
            Door.Status.CLOSED,
            Door.Status.LOCKED,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            0.0,
        ),
        (
            Door.Status.LOCKED,
            Door.Status.OPEN,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            -1.5,
        ),
        (
            Door.Status.LOCKED,
            Door.Status.CLOSED,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            0.0,
        ),
        (
            Door.Status.LOCKED,
            Door.Status.LOCKED,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            0.0,
        ),
        (
            None,
            Door.Status.OPEN,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            0,
        ),
        (
            Door.Status.OPEN,
            None,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            0,
        ),
        (
            None,
            None,
            Action.ACTUATE,
            {'reward_open': -1.5, 'reward_close': 1.5},
            0,
        ),
    ],
)
def test_actuate_door(
    door_status: Door.Status,
    next_door_status: Door.Status,
    action: Action,
    kwargs,
    expected: float,
):
    state = make_door_state(door_status)
    next_state = make_door_state(next_door_status)
    assert actuate_door(state, action, next_state, **kwargs) == expected


@pytest.mark.parametrize(
    'state,next_state,kwargs,expected',
    [
        # default values
        (
            make_key_state(False),
            make_key_state(False),
            {'object_type': Key},
            0.0,
        ),
        (
            make_key_state(False),
            make_key_state(True),
            {'object_type': Key},
            1.0,
        ),
        (
            make_key_state(True),
            make_key_state(False),
            {'object_type': Key},
            -1.0,
        ),
        (make_key_state(True), make_key_state(True), {'object_type': Key}, 0.0),
        # custom values
        (
            make_key_state(False),
            make_key_state(False),
            {'object_type': Key, 'reward_pick': -1.5, 'reward_drop': 1.5},
            0.0,
        ),
        (
            make_key_state(False),
            make_key_state(True),
            {'object_type': Key, 'reward_pick': -1.5, 'reward_drop': 1.5},
            -1.5,
        ),
        (
            make_key_state(True),
            make_key_state(False),
            {'object_type': Key, 'reward_pick': -1.5, 'reward_drop': 1.5},
            1.5,
        ),
        (
            make_key_state(True),
            make_key_state(True),
            {'object_type': Key, 'reward_pick': -1.5, 'reward_drop': 1.5},
            0.0,
        ),
        # wrong object
        (
            make_key_state(False),
            make_key_state(False),
            {'object_type': Wall},
            0.0,
        ),
        (
            make_key_state(False),
            make_key_state(True),
            {'object_type': Wall},
            0.0,
        ),
        (
            make_key_state(True),
            make_key_state(False),
            {'object_type': Wall},
            0.0,
        ),
        (
            make_key_state(True),
            make_key_state(True),
            {'object_type': Wall},
            0.0,
        ),
    ],
)
def test_pickndrop(
    state: State,
    next_state: State,
    kwargs,
    expected: float,
    forbidden_action_maker,
):
    action = forbidden_action_maker()
    assert pickndrop(state, action, next_state, **kwargs) == expected


# TODO: test `reduce` function


@pytest.mark.parametrize(
    'name,kwargs',
    [
        ('reduce_sum', {'reward_functions': []}),
        (
            'overlap',
            {'object_type': Exit, 'reward_on': 1.0, 'reward_off': -1.0},
        ),
        ('living_reward', {'reward': 1.0}),
        ('reach_exit', {'reward_on': 1.0, 'reward_off': -1.0}),
        ('bump_moving_obstacle', {'reward': -1.0}),
        (
            'proportional_to_distance',
            {
                'distance_function': Position.manhattan_distance,
                'object_type': Exit,
                'reward_per_unit_distance': -0.1,
            },
        ),
        (
            'getting_closer',
            {
                'distance_function': Position.manhattan_distance,
                'object_type': Exit,
                'reward_closer': -0.1,
                'reward_further': -0.1,
            },
        ),
        (
            'getting_closer_shortest_path',
            {
                'object_type': Exit,
                'reward_closer': -0.1,
                'reward_further': -0.1,
            },
        ),
        ('bump_into_wall', {'reward': -1.0}),
        (
            'actuate_door',
            {
                'reward_open': 1.0,
                'reward_close': -1.0,
            },
        ),
        (
            'pickndrop',
            {
                'object_type': Key,
                'reward_pick': 1.0,
                'reward_drop': -1.0,
            },
        ),
    ],
)
def test_factory_valid(name: str, kwargs):
    factory(name, **kwargs)


@pytest.mark.parametrize(
    'name,kwargs,exception',
    [
        ('invalid', {}, ValueError),
        ('reduce_sum', {}, ValueError),
        ('overlap', {}, ValueError),
        ('proportional_to_distance', {}, ValueError),
        ('getting_closer', {}, ValueError),
        ('getting_closer_shortest_path', {}, ValueError),
        ('pickndrop', {}, ValueError),
    ],
)
def test_factory_invalid(name: str, kwargs, exception: Type[Exception]):
    with pytest.raises(exception):
        factory(name, **kwargs)
