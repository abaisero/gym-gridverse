import pytest

from gym_gridverse.action import Action
from gym_gridverse.envs.reward_functions import (
    actuate_door,
    bump_into_wall,
    bump_moving_obstacle,
    factory,
    getting_closer,
    living_reward,
    pickndrop,
    proportional_to_distance,
    reach_goal,
)
from gym_gridverse.geometry import Orientation, Position, PositionOrTuple
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Goal,
    Key,
    MovingObstacle,
    Wall,
)
from gym_gridverse.info import Agent, Grid
from gym_gridverse.state import State


def make_5x5_goal_state() -> State:
    """makes a simple 5x5 state with goal object in the middle"""
    grid = Grid(5, 5)
    grid[2, 2] = Goal()
    agent = Agent((0, 0), Orientation.N)
    return State(grid, agent)


def make_goal_state(agent_on_goal: bool) -> State:
    """makes a simple state with a wall in front of the agent"""
    grid = Grid(2, 1)
    grid[0, 0] = Goal()
    agent_position = (0, 0) if agent_on_goal else (1, 0)
    agent = Agent(agent_position, Orientation.N)
    return State(grid, agent)


def make_wall_state(orientation: Orientation = Orientation.N) -> State:
    """makes a simple state with goal object and agent on or off the goal"""
    grid = Grid(2, 1)
    grid[0, 0] = Wall()
    agent = Agent((1, 0), orientation)
    return State(grid, agent)


def make_door_state(door_status: Door.Status) -> State:
    """makes a simple state with a door"""
    grid = Grid(2, 1)
    grid[0, 0] = Door(door_status, Colors.RED)
    agent = Agent((1, 0), Orientation.N)
    return State(grid, agent)


def make_key_state(has_key: bool) -> State:
    """makes a simple state with a door"""
    grid = Grid(1, 1)
    obj = Key(Colors.RED) if has_key else None
    agent = Agent((0, 0), Orientation.N, obj)
    return State(grid, agent)


def make_moving_obstacle_state(agent_on_obstacle: bool) -> State:
    """makes a simple state with goal object and agent on or off the goal"""
    grid = Grid(2, 1)
    grid[0, 0] = MovingObstacle()
    agent_position = (0, 0) if agent_on_obstacle else (1, 0)
    agent = Agent(agent_position, Orientation.N)
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
    'agent_on_goal,kwargs,expected',
    [
        (True, {}, 1.0),
        (False, {}, 0.0),
        (True, {'reward_on': 10.0, 'reward_off': -1.0}, 10.0),
        (False, {'reward_on': 10.0, 'reward_off': -1.0}, -1.0),
    ],
)
def test_reach_goal(
    agent_on_goal: bool,
    kwargs,
    expected: float,
    forbidden_state_maker,
    forbidden_action_maker,
):
    state = forbidden_state_maker()
    action = forbidden_action_maker()
    next_state = make_goal_state(agent_on_goal=agent_on_goal)
    assert reach_goal(state, action, next_state, **kwargs) == expected


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
        ((0, 0), {}, -4.0),
        ((0, 1), {}, -3.0),
        ((0, 2), {}, -2.0),
        ((0, 3), {}, -3.0),
        ((0, 4), {}, -4.0),
        # moving agent on the middle row
        ((2, 0), {}, -2.0),
        ((2, 1), {}, -1.0),
        ((2, 2), {}, 0.0),
        ((2, 3), {}, -1.0),
        ((2, 4), {}, -2.0),
        # moving agent on the top row
        ((0, 0), {'reward_per_unit_distance': 0.1}, 0.40),
        ((0, 1), {'reward_per_unit_distance': 0.1}, 0.30),
        ((0, 2), {'reward_per_unit_distance': 0.1}, 0.20),
        ((0, 3), {'reward_per_unit_distance': 0.1}, 0.30),
        ((0, 4), {'reward_per_unit_distance': 0.1}, 0.40),
        # moving agent on the middle row
        ((2, 0), {'reward_per_unit_distance': 0.1}, 0.20),
        ((2, 1), {'reward_per_unit_distance': 0.1}, 0.10),
        ((2, 2), {'reward_per_unit_distance': 0.1}, 0.0),
        ((2, 3), {'reward_per_unit_distance': 0.1}, 0.10),
        ((2, 4), {'reward_per_unit_distance': 0.1}, 0.20),
    ],
)
def test_proportional_to_distance_default(
    position: PositionOrTuple,
    kwargs,
    expected: float,
    forbidden_state_maker,
    forbidden_action_maker,
):
    state = forbidden_state_maker()
    action = forbidden_action_maker()
    next_state = make_5x5_goal_state()
    # TODO find better way to construct this state
    next_state.agent.position = Position.from_position_or_tuple(position)

    reward = proportional_to_distance(
        state, action, next_state, object_type=Goal, **kwargs
    )
    assert round(reward, 7) == expected


@pytest.mark.parametrize(
    'agent_on_goal,next_agent_on_goal,kwargs,expected',
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
    agent_on_goal: bool,
    next_agent_on_goal: bool,
    kwargs,
    expected: float,
    forbidden_action_maker,
):
    state = make_goal_state(agent_on_goal)
    action = forbidden_action_maker()
    next_state = make_goal_state(next_agent_on_goal)
    assert (
        getting_closer(state, action, next_state, object_type=Goal, **kwargs)
        == expected
    )


@pytest.mark.parametrize(
    'state,action,kwargs,expected',
    [
        # not bumping
        (make_wall_state(), Action.MOVE_LEFT, {}, 0.0),
        (make_wall_state(), Action.PICK_N_DROP, {}, 0.0),
        # bumping
        (make_wall_state(Orientation.N), Action.MOVE_FORWARD, {}, -1.0),
        (make_wall_state(Orientation.E), Action.MOVE_LEFT, {}, -1.0),
        (make_wall_state(Orientation.S), Action.MOVE_BACKWARD, {}, -1.0),
        (make_wall_state(Orientation.W), Action.MOVE_RIGHT, {}, -1.0),
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


@pytest.mark.parametrize(
    'name,kwargs',
    [
        ('chain', {'reward_functions': []}),
        (
            'overlap',
            {'object_type': Goal, 'reward_on': 1.0, 'reward_off': -1.0},
        ),
        ('living_reward', {'reward': 1.0}),
        ('reach_goal', {'reward_on': 1.0, 'reward_off': -1.0}),
        ('bump_moving_obstacle', {'reward': -1.0}),
        (
            'proportional_to_distance',
            {
                'distance_function': Position.manhattan_distance,
                'object_type': Goal,
                'reward_per_unit_distance': -0.1,
            },
        ),
        (
            'getting_closer',
            {
                'distance_function': Position.manhattan_distance,
                'object_type': Goal,
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
        ('chain', {}, ValueError),
        ('overlap', {}, ValueError),
        ('living_reward', {}, ValueError),
        ('reach_goal', {}, ValueError),
        ('bump_moving_obstacle', {}, ValueError),
        ('proportional_to_distance', {}, ValueError),
        ('getting_closer', {}, ValueError),
        ('bump_into_wall', {}, ValueError),
        ('actuate_door', {}, ValueError),
        ('pickndrop', {}, ValueError),
    ],
)
def test_factory_invalid(name: str, kwargs, exception: Exception):
    with pytest.raises(exception):  # type: ignore
        factory(name, **kwargs)
