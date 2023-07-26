import glob
from typing import List, Optional, Sequence, Type, Union

import pytest

from gym_gridverse.action import Action
from gym_gridverse.agent import Agent
from gym_gridverse.envs.yaml import factory as yaml_factory
from gym_gridverse.geometry import Area, Orientation, Position, Shape
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import (
    Color,
    Door,
    Exit,
    Floor,
    GridObject,
    Key,
    Wall,
)
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import (
    ActionSpace,
    ObservationSpace,
    _max_color_index,
    _max_object_status,
    _max_object_type,
)


# TODO: bad test;  implementation detail
@pytest.mark.parametrize(
    'colors,expected',
    [
        ([Color.NONE], Color.NONE.value),
        ([Color.NONE, Color.RED], Color.RED.value),
        ([Color.NONE, Color.RED, Color.GREEN], Color.GREEN.value),
        (
            [Color.NONE, Color.RED, Color.GREEN, Color.BLUE],
            Color.BLUE.value,
        ),
        (
            [Color.NONE, Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW],
            Color.YELLOW.value,
        ),
    ],
)
def test_max_color_index(colors: Sequence[Color], expected: int):
    assert _max_color_index(colors) == expected


# TODO: bad test;  implementation detail
@pytest.mark.parametrize(
    'object_types,expected',
    [
        ([Floor, Exit], Floor.num_states()),
        ([Floor, Exit, Door], Door.num_states()),
    ],
)
def test_max_object_status(
    object_types: Sequence[Type[GridObject]], expected: int
):
    assert _max_object_status(object_types) == expected


# TODO: bad test;  implementation detail
@pytest.mark.parametrize(
    'object_types,expected',
    [
        ([Floor, Exit], Exit.type_index()),
        ([Floor, Exit, Door], Door.type_index()),
    ],
)
def test_max_object_type(
    object_types: Sequence[Type[GridObject]], expected: int
):
    assert _max_object_type(object_types) == expected


@pytest.mark.parametrize(
    'action_space,expected_contains,expected_not_contains',
    [
        (
            ActionSpace(list(Action)),
            [
                Action.MOVE_FORWARD,
                Action.MOVE_BACKWARD,
                Action.MOVE_LEFT,
                Action.MOVE_RIGHT,
                Action.TURN_LEFT,
                Action.TURN_RIGHT,
                Action.ACTUATE,
                Action.PICK_N_DROP,
            ],
            [],
        ),
        (
            ActionSpace(
                [
                    Action.MOVE_FORWARD,
                    Action.MOVE_BACKWARD,
                    Action.MOVE_LEFT,
                    Action.MOVE_RIGHT,
                ]
            ),
            [
                Action.MOVE_FORWARD,
                Action.MOVE_BACKWARD,
                Action.MOVE_LEFT,
                Action.MOVE_RIGHT,
            ],
            [
                Action.TURN_LEFT,
                Action.TURN_RIGHT,
                Action.ACTUATE,
                Action.PICK_N_DROP,
            ],
        ),
    ],
)
def test_action_space_contains(
    action_space: ActionSpace,
    expected_contains: Sequence[Action],
    expected_not_contains: Sequence[Action],
):
    assert action_space.num_actions == len(expected_contains)

    for action in expected_contains:
        assert action_space.contains(action)

    for action in expected_not_contains:
        assert not action_space.contains(action)


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
        (Shape(2, 5), Position(1, 2)),
        (Shape(3, 5), Position(2, 2)),
        (Shape(2, 7), Position(1, 3)),
        (Shape(3, 7), Position(2, 3)),
    ],
)
def test_observation_space_agent_position(shape: Shape, expected: Position):
    observation_space = ObservationSpace(shape, [], [])
    assert observation_space.agent_position == expected


def space_contains_observation(
    space_shape: Shape = Shape(2, 5),
    space_objs: Sequence[Type[GridObject]] = [Floor],
    space_colors: Sequence[Color] = [],
    grid: Grid = Grid.from_shape((2, 5)),
    agent_obj: Union[GridObject, None] = None,
    agent_pos: Position = Position(0, 0),
    orientation: Orientation = Orientation.F,
):
    """helper function to test whether space contains obs given inputs"""
    observation_space = ObservationSpace(space_shape, space_objs, space_colors)
    obs = Observation(grid, Agent(agent_pos, orientation, agent_obj))

    return observation_space.contains(obs)


@pytest.mark.parametrize(
    'space_shape,observation_shape,agent_position,expected',
    [
        (Shape(2, 3), Shape(2, 3), Position(1, 1), True),
        (Shape(4, 5), Shape(4, 5), Position(3, 2), True),
        # invalid
        (Shape(2, 3), Shape(2, 5), Position(1, 2), False),
        (Shape(2, 3), Shape(3, 3), Position(2, 1), False),
        (Shape(4, 5), Shape(4, 7), Position(3, 3), False),
        (Shape(4, 5), Shape(5, 5), Position(4, 2), False),
    ],
)
def test_observation_space_contains__shape(
    space_shape: Shape,
    observation_shape: Shape,
    agent_position: Position,
    expected: bool,
):
    observation_space = ObservationSpace(space_shape, [Floor], [Color.NONE])
    observation = Observation(
        Grid.from_shape(observation_shape),
        Agent(agent_position, Orientation.F),
    )

    assert observation_space.contains(observation) == expected


@pytest.mark.parametrize(
    'space_object_types,observation_objects,agent_grid_object,expected',
    [
        ([Floor], [[Floor(), Floor(), Floor()]], None, True),
        ([Floor, Wall], [[Floor(), Floor(), Floor()]], None, True),
        ([Floor, Wall], [[Floor(), Floor(), Wall()]], None, True),
        ([Floor, Wall], [[Floor(), Floor(), Floor()]], Wall(), True),
        # invalid
        ([Floor], [[Floor(), Floor(), Wall()]], None, False),
        ([Floor], [[Floor(), Floor(), Wall()]], Wall(), False),
    ],
)
def test_observation_space_contains__object_types(
    space_object_types: Sequence[Type[GridObject]],
    observation_objects: List[List[GridObject]],
    agent_grid_object: Optional[GridObject],
    expected: bool,
):
    # NOTE:  observation_objects should have shape (1, 3)
    observation_space = ObservationSpace(
        Shape(1, 3), space_object_types, [Color.NONE]
    )
    observation = Observation(
        Grid(observation_objects),
        Agent(Position(0, 1), Orientation.F, agent_grid_object),
    )

    assert observation_space.contains(observation) == expected


@pytest.mark.parametrize(
    'space_colors,observation_objects,agent_grid_object,expected',
    [
        ([Color.RED], [[Key(Color.RED)], [Key(Color.RED)]], None, True),
        (
            [Color.RED, Color.BLUE],
            [[Key(Color.RED)], [Key(Color.BLUE)]],
            None,
            True,
        ),
        (
            [Color.RED, Color.BLUE],
            [[Key(Color.RED)], [Key(Color.RED)]],
            Key(Color.BLUE),
            True,
        ),
        # invalid
        ([Color.RED], [[Key(Color.RED)], [Key(Color.BLUE)]], None, False),
        (
            [Color.RED],
            [[Key(Color.RED)], [Key(Color.RED)]],
            Key(Color.BLUE),
            False,
        ),
    ],
)
def test_observation_space_contains__colors(
    space_colors: Sequence[Color],
    observation_objects: List[List[GridObject]],
    agent_grid_object: Optional[GridObject],
    expected: bool,
):
    # NOTE:  observation_objects should have shape (2, 1)
    observation_space = ObservationSpace(Shape(2, 1), [Key], space_colors)
    observation = Observation(
        Grid(observation_objects),
        Agent(Position(1, 0), Orientation.F, agent_grid_object),
    )

    assert observation_space.contains(observation) == expected


@pytest.mark.parametrize(
    'shape,position,position_ok',
    [
        (Shape(1, 3), Position(0, 1), True),
        (Shape(2, 5), Position(1, 2), True),
        # invalid
        (Shape(1, 3), Position(-1, -1), False),
        (Shape(1, 3), Position(1, 3), False),
        (Shape(2, 5), Position(-1, -1), False),
        (Shape(2, 5), Position(2, 5), False),
    ],
)
@pytest.mark.parametrize(
    'orientation,orientation_ok',
    [
        (Orientation.F, True),
        # all orientations are valid now
        (Orientation.B, True),
        (Orientation.R, True),
        (Orientation.L, True),
    ],
)
def test_observation_space_contains__agent_transform(
    shape: Shape,
    position: Position,
    position_ok: bool,
    orientation: Orientation,
    orientation_ok: bool,
):
    observation_space = ObservationSpace(shape, [Floor], [Color.NONE])
    observation = Observation(
        Grid.from_shape(shape), Agent(position, orientation)
    )

    expected = position_ok and orientation_ok
    assert observation_space.contains(observation) == expected


@pytest.mark.parametrize(
    'shape',
    [
        Shape(2, 3),
        Shape(4, 5),
    ],
)
@pytest.mark.parametrize(
    'object_types',
    [
        [Floor],
        [Floor, Wall],
    ],
)
@pytest.mark.parametrize(
    'colors',
    [
        [Color.NONE],
        [Color.NONE, Color.RED],
    ],
)
@pytest.mark.parametrize(
    'position',
    [
        Position(0, 0),
        Position(0, 1),
        Position(1, 0),
        Position(1, 1),
    ],
)
@pytest.mark.parametrize(
    'orientation',
    [Orientation.F],
)
def test_observation_space_contains(
    shape: Shape,
    object_types: Sequence[Type[GridObject]],
    colors: Sequence[Color],
    position: Position,
    orientation: Orientation,
):
    observation_space = ObservationSpace(shape, object_types, colors)
    observation = Observation(
        Grid.from_shape(shape), Agent(position, orientation)
    )

    assert observation_space.contains(observation)


# NOTE testing of Space.contains methods for all yaml files in yaml/
@pytest.mark.parametrize('path', glob.glob('yaml/*.yaml'))
def test_space_contains_from_yaml(path: str):
    env = yaml_factory.factory_env_from_yaml(path)

    state = env.functional_reset()
    env.state_space.contains(state)

    observation = env.functional_observation(state)
    env.observation_space.contains(observation)
