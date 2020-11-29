from typing import Sequence, Type

import pytest

from gym_gridverse.actions import Actions
from gym_gridverse.geometry import (
    Area,
    Orientation,
    Position,
    PositionOrTuple,
    Shape,
)
from gym_gridverse.grid_object import (
    Colors,
    Door,
    Floor,
    Goal,
    GridObject,
    Key,
    NoneGridObject,
)
from gym_gridverse.info import Agent, Grid
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import (
    ActionSpace,
    ObservationSpace,
    _max_color_index,
    _max_object_status,
    _max_object_type,
)


def test_max_color_index():
    colors = [Colors.RED, Colors.BLUE]
    assert _max_color_index(colors) == Colors.BLUE.value


@pytest.mark.parametrize(
    'object_types,expected',
    [([Floor, Goal], 0), ([Door, Goal], len(Door.Status))],
)
def test_max_object_status(
    object_types: Sequence[Type[GridObject]], expected: int
):
    assert _max_object_status(object_types) == expected


@pytest.mark.parametrize(
    'object_types,expected',
    [([Floor, Goal], Goal.type_index), ([Door, Goal], Door.type_index)],
)
def test_max_object_type(
    object_types: Sequence[Type[GridObject]], expected: int
):
    assert _max_object_type(object_types) == expected


@pytest.mark.parametrize(
    'action_space,expected_contains,expected_not_contains',
    [
        (
            ActionSpace(list(Actions)),
            [
                Actions.MOVE_FORWARD,
                Actions.MOVE_BACKWARD,
                Actions.MOVE_LEFT,
                Actions.MOVE_RIGHT,
                Actions.TURN_LEFT,
                Actions.TURN_RIGHT,
                Actions.ACTUATE,
                Actions.PICK_N_DROP,
            ],
            [],
        ),
        (
            ActionSpace(
                [
                    Actions.MOVE_FORWARD,
                    Actions.MOVE_BACKWARD,
                    Actions.MOVE_LEFT,
                    Actions.MOVE_RIGHT,
                ]
            ),
            [
                Actions.MOVE_FORWARD,
                Actions.MOVE_BACKWARD,
                Actions.MOVE_LEFT,
                Actions.MOVE_RIGHT,
            ],
            [
                Actions.TURN_LEFT,
                Actions.TURN_RIGHT,
                Actions.ACTUATE,
                Actions.PICK_N_DROP,
            ],
        ),
    ],
)
def test_action_space_contains(
    action_space: ActionSpace,
    expected_contains: Sequence[Actions],
    expected_not_contains: Sequence[Actions],
):
    for action in expected_contains:
        assert action_space.contains(action)

    for action in expected_not_contains:
        assert not action_space.contains(action)


def test_action_space_num_actions():
    action_space = ActionSpace(list(Actions))
    assert action_space.num_actions == len(Actions)

    action_space = ActionSpace(
        [
            Actions.MOVE_FORWARD,
            Actions.MOVE_BACKWARD,
            Actions.MOVE_LEFT,
            Actions.MOVE_RIGHT,
        ]
    )
    assert action_space.num_actions == 4


@pytest.mark.parametrize(
    'shape,expected',
    [
        (Shape(2, 5), Area((-1, 0), (-2, 2))),
        (Shape(3, 5), Area((-2, 0), (-2, 2))),
        (Shape(2, 7), Area((-1, 0), (-3, 3))),
        (Shape(3, 7), Area((-2, 0), (-3, 3))),
    ],
)
def test_observation_space_area(shape: Shape, expected: Area):
    observation_space = ObservationSpace(shape, [], [])
    assert observation_space.area == expected


@pytest.mark.parametrize(
    'shape,expected',
    [
        (Shape(2, 5), (1, 2)),
        (Shape(3, 5), (2, 2)),
        (Shape(2, 7), (1, 3)),
        (Shape(3, 7), (2, 3)),
    ],
)
def test_observation_space_agent_position(
    shape: Shape, expected: PositionOrTuple
):
    observation_space = ObservationSpace(shape, [], [])
    assert observation_space.agent_position == expected


def space_contains_observation(
    space_shape: Shape = Shape(2, 5),
    space_objs: Sequence[Type[GridObject]] = [Floor],
    space_colors: Sequence[Colors] = [],
    grid: Grid = Grid(2, 5),
    agent_obj: GridObject = None,
    agent_pos: Position = Position(0, 0),
    orientation: Orientation = Orientation.N,
):
    """helper function to test whether space contains obs given inputs """
    observation_space = ObservationSpace(space_shape, space_objs, space_colors)
    obs = Observation(grid, Agent(agent_pos, orientation, agent_obj))

    return observation_space.contains(obs)


@pytest.mark.parametrize(
    'space_shape,obs_shape,contains',
    [
        (Shape(2, 5), Shape(2, 5), True),
        (Shape(2, 5), Shape(2, 7), False),
        (Shape(2, 3), Shape(2, 5), False),
        (Shape(3, 1), Shape(3, 1), True),
        (Shape(2, 1), Shape(3, 1), False),
        (Shape(3, 1), Shape(2, 1), False),
    ],
)
def test_observation_space_contains_grid_shape(
    space_shape: Shape, obs_shape: Shape, contains: bool
):
    assert (
        space_contains_observation(
            space_shape=space_shape,
            grid=Grid(*obs_shape),
        )
        == contains
    )


@pytest.mark.parametrize(
    'space_colors,obj_color,contains',
    [
        ([], Colors.NONE, True),
        ([], Colors.YELLOW, False),
        ([Colors.YELLOW], Colors.YELLOW, True),
        ([Colors.YELLOW], Colors.NONE, True),
        ([Colors.YELLOW], Colors.BLUE, False),
    ],
)
def test_observation_space_contains_colors(
    space_colors: Sequence[Colors], obj_color: Colors, contains: bool
):
    assert (
        space_contains_observation(
            space_objs=[Floor, Key],
            space_colors=space_colors,
            agent_obj=Key(obj_color),
        )
        == contains
    )


@pytest.mark.parametrize(
    'space_objs,agent_obj,contains',
    [
        ([Floor], NoneGridObject(), True),
        ([Floor], Key(Colors.BLUE), False),
        ([Floor, Key], Key(Colors.BLUE), True),
        ([Floor, Key], NoneGridObject(), True),
    ],
)
def test_observation_space_contains_agent_object_type(
    space_objs: Sequence[Type[GridObject]],
    agent_obj: GridObject,
    contains: bool,
):
    space_colors = [agent_obj.color]
    assert (
        space_contains_observation(
            space_objs=space_objs,
            space_colors=space_colors,
            agent_obj=agent_obj,
        )
        == contains
    )


@pytest.mark.parametrize(
    'agent_pos,contains',
    [
        (Position(0, 0), True),
        (Position(0, -1), False),
        (Position(-1, 0), False),
        (Position(-1, -1), False),
        (Position(10, 0), False),
        (Position(10, -1), False),
    ],
)
def test_observation_space_position_in_grid(
    agent_pos: Position, contains: bool
):
    assert space_contains_observation(agent_pos=agent_pos) == contains


@pytest.mark.parametrize(
    'space_objs,space_colors,obj_in_grid,contains',
    [
        ([Floor], [], Floor(), True),
        ([Floor], [Colors.BLUE], Key(Colors.BLUE), False),
        ([Floor, Key], [Colors.BLUE], Key(Colors.BLUE), True),
        ([Floor, Key], [], Floor(), True),
    ],
)
def test_observation_space_contains_object_type(
    space_objs: Sequence[Type[GridObject]],
    space_colors: Sequence[Colors],
    obj_in_grid: GridObject,
    contains: bool,
):
    grid = Grid(2, 5)
    grid[0, 1] = obj_in_grid

    assert (
        space_contains_observation(
            grid=grid, space_objs=space_objs, space_colors=space_colors
        )
        == contains
    )


@pytest.mark.parametrize(
    'orientation,contains',
    [
        (Orientation.N, True),
        (Orientation.S, False),
        (Orientation.E, False),
        (Orientation.W, False),
    ],
)
def test_observation_space_contains_orientation(
    orientation: Orientation, contains: bool
):
    assert space_contains_observation(orientation=orientation) == contains
