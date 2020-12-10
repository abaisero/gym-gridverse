import glob
from typing import Optional, Sequence, Type

import pytest

from gym_gridverse.actions import Actions
from gym_gridverse.envs.yaml import factory as yaml_factory
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
    Wall,
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


# TODO bad test;  implementation detail
@pytest.mark.parametrize(
    'colors,expected',
    [
        ([Colors.NONE], Colors.NONE.value),
        ([Colors.NONE, Colors.RED], Colors.RED.value),
        ([Colors.NONE, Colors.RED, Colors.GREEN], Colors.GREEN.value),
        (
            [Colors.NONE, Colors.RED, Colors.GREEN, Colors.BLUE],
            Colors.BLUE.value,
        ),
        (
            [Colors.NONE, Colors.RED, Colors.GREEN, Colors.BLUE, Colors.YELLOW],
            Colors.YELLOW.value,
        ),
    ],
)
def test_max_color_index(colors: Sequence[Colors], expected: int):
    assert _max_color_index(colors) == expected


# TODO bad test;  implementation detail
@pytest.mark.parametrize(
    'object_types,expected',
    [
        ([Floor, Goal], Floor.num_states()),
        ([Floor, Goal, Door], Door.num_states()),
    ],
)
def test_max_object_status(
    object_types: Sequence[Type[GridObject]], expected: int
):
    assert _max_object_status(object_types) == expected


# TODO bad test;  implementation detail
@pytest.mark.parametrize(
    'object_types,expected',
    [
        ([Floor, Goal], Goal.type_index),
        ([Floor, Goal, Door], Door.type_index),
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
    'space_shape,observation_shape,agent_position,expected',
    [
        (Shape(2, 3), Shape(2, 3), (1, 1), True),
        (Shape(4, 5), Shape(4, 5), (3, 2), True),
        # invalid
        (Shape(2, 3), Shape(2, 5), (1, 2), False),
        (Shape(2, 3), Shape(3, 3), (2, 1), False),
        (Shape(4, 5), Shape(4, 7), (3, 3), False),
        (Shape(4, 5), Shape(5, 5), (4, 2), False),
    ],
)
def test_observation_space_contains__shape(
    space_shape: Shape,
    observation_shape: Shape,
    agent_position: PositionOrTuple,
    expected: bool,
):
    observation_space = ObservationSpace(space_shape, [Floor], [Colors.NONE])
    observation = Observation(
        Grid(observation_shape.height, observation_shape.width),
        Agent(agent_position, Orientation.N),
    )

    assert observation_space.contains(observation) == expected


@pytest.mark.parametrize(
    'space_object_types,observation_objects,agent_object,expected',
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
    observation_objects: Sequence[Sequence[GridObject]],
    agent_object: Optional[GridObject],
    expected: bool,
):
    # NOTE:  observation_objects should have shape (1, 3)
    observation_space = ObservationSpace(
        Shape(1, 3), space_object_types, [Colors.NONE]
    )
    observation = Observation(
        Grid.from_objects(observation_objects),
        Agent((0, 1), Orientation.N, agent_object),
    )

    assert observation_space.contains(observation) == expected


@pytest.mark.parametrize(
    'space_colors,observation_objects,agent_object,expected',
    [
        ([Colors.RED], [[Key(Colors.RED)], [Key(Colors.RED)]], None, True),
        (
            [Colors.RED, Colors.BLUE],
            [[Key(Colors.RED)], [Key(Colors.BLUE)]],
            None,
            True,
        ),
        (
            [Colors.RED, Colors.BLUE],
            [[Key(Colors.RED)], [Key(Colors.RED)]],
            Key(Colors.BLUE),
            True,
        ),
        # invalid
        ([Colors.RED], [[Key(Colors.RED)], [Key(Colors.BLUE)]], None, False),
        (
            [Colors.RED],
            [[Key(Colors.RED)], [Key(Colors.RED)]],
            Key(Colors.BLUE),
            False,
        ),
    ],
)
def test_observation_space_contains__colors(
    space_colors: Sequence[Colors],
    observation_objects: Sequence[Sequence[GridObject]],
    agent_object: Optional[GridObject],
    expected: bool,
):
    # NOTE:  observation_objects should have shape (2, 1)
    observation_space = ObservationSpace(Shape(2, 1), [Key], space_colors)
    observation = Observation(
        Grid.from_objects(observation_objects),
        Agent((1, 0), Orientation.N, agent_object),
    )

    assert observation_space.contains(observation) == expected


@pytest.mark.parametrize(
    'shape,position,position_ok',
    [
        (Shape(1, 3), (0, 1), True),
        (Shape(2, 5), (1, 2), True),
        # invalid
        (Shape(1, 3), (-1, -1), False),
        (Shape(1, 3), (1, 3), False),
        (Shape(2, 5), (-1, -1), False),
        (Shape(2, 5), (2, 5), False),
    ],
)
@pytest.mark.parametrize(
    'orientation,orientation_ok',
    [
        (Orientation.N, True),
        # invalid
        (Orientation.S, False),
        (Orientation.E, False),
        (Orientation.W, False),
    ],
)
def test_observation_space_contains__agent_pose(
    shape: Shape,
    position: PositionOrTuple,
    position_ok: bool,
    orientation: Orientation,
    orientation_ok: bool,
):
    observation_space = ObservationSpace(shape, [Floor], [Colors.NONE])
    observation = Observation(
        Grid(shape.height, shape.width), Agent(position, orientation)
    )

    expected = position_ok and orientation_ok
    assert observation_space.contains(observation) == expected


@pytest.mark.parametrize('shape', [Shape(2, 3), Shape(4, 5)])
@pytest.mark.parametrize('object_types', [[Floor], [Floor, Wall]])
@pytest.mark.parametrize('colors', [[Colors.NONE], [Colors.NONE, Colors.RED]])
@pytest.mark.parametrize('position', [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize('orientation', [Orientation.N])
def test_observation_space_contains(
    shape: Shape,
    object_types: Sequence[Type[GridObject]],
    colors: Sequence[Colors],
    position: PositionOrTuple,
    orientation: Orientation,
):
    observation_space = ObservationSpace(shape, object_types, colors)
    observation = Observation(
        Grid(shape.height, shape.width),
        Agent(position, orientation),
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
