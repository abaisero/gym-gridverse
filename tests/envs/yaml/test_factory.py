import glob

import pytest
from schema import SchemaError

import gym_gridverse.envs.yaml.factory as yaml_factory
from gym_gridverse.actions import Actions
from gym_gridverse.envs import InnerEnv
from gym_gridverse.geometry import Shape
from gym_gridverse.grid_object import Colors, GridObject
from gym_gridverse.spaces import ActionSpace, ObservationSpace, StateSpace


@pytest.mark.parametrize(
    'data',
    [
        [5, 11],
        [11, 5],
    ],
)
def test_factory_shape(data):
    shape = yaml_factory.factory_shape(data)
    assert isinstance(shape, Shape)


@pytest.mark.parametrize(
    'data',
    [
        # non-positive
        [0, 1],
        [1, 0],
        # wrong length
        [],
        [1],
        [1, 1, 1],
    ],
)
def test_factory_shape_fail(data):
    with pytest.raises(SchemaError):
        yaml_factory.factory_shape(data)


@pytest.mark.parametrize(
    'data',
    [
        [1, 2],
        [2, 1],
    ],
)
def test_factory_layout(data):
    layout = yaml_factory.factory_layout(data)
    assert isinstance(layout, tuple)
    assert len(layout) == 2
    assert isinstance(layout[0], int) and isinstance(layout[1], int)


@pytest.mark.parametrize(
    'data',
    [
        # non-positive
        [0, 1],
        [1, 0],
        # wrong length
        [],
        [1],
        [1, 1, 1],
    ],
)
def test_factory_layout_fail(data):
    with pytest.raises(SchemaError):
        yaml_factory.factory_layout(data)


@pytest.mark.parametrize(
    'data',
    [
        {'shape': [11, 11], 'objects': ['Floor'], 'colors': ['NONE']},
    ],
)
def test_factory_state_space(data):
    state_space = yaml_factory.factory_state_space(data)
    assert isinstance(state_space, StateSpace)
    assert isinstance(state_space.grid_shape, Shape)

    for object_type in state_space.object_types:
        assert issubclass(object_type, GridObject)

    for color in state_space.colors:
        assert isinstance(color, Colors)


@pytest.mark.parametrize(
    'data',
    [
        # missing fields
        {'objects': ['Floor'], 'colors': ['NONE']},
        {'shape': [11, 11], 'colors': ['NONE']},
        {'shape': [11, 11], 'objects': ['Floor']},
    ],
)
def test_factory_state_space_fail(data):
    with pytest.raises(SchemaError):
        yaml_factory.factory_state_space(data)


@pytest.mark.parametrize(
    'data',
    [
        ['MOVE_FORWARD', 'MOVE_BACKWARD'],
    ],
)
def test_factory_action_space(data):
    action_space = yaml_factory.factory_action_space(data)
    assert isinstance(action_space, ActionSpace)
    assert len(set(action_space.actions)) == len(action_space.actions)

    for action in action_space.actions:
        isinstance(action, Actions)


@pytest.mark.parametrize(
    'data',
    [
        [],
        ['MOVE_FORWARD'],
        ['MOVE_FORWARD', 'MOVE_FORWARD'],
    ],
)
def test_factory_action_space_fail(data):
    with pytest.raises(SchemaError):
        yaml_factory.factory_action_space(data)


@pytest.mark.parametrize(
    'data',
    [
        {'shape': [5, 5], 'objects': ['Floor'], 'colors': ['NONE']},
    ],
)
def test_factory_observation_space(data):
    observation_space = yaml_factory.factory_observation_space(data)
    assert isinstance(observation_space, ObservationSpace)

    for object_type in observation_space.object_types:
        assert issubclass(object_type, GridObject)

    for color in observation_space.colors:
        assert isinstance(color, Colors)


@pytest.mark.parametrize(
    'data',
    [
        # invalid (even width)
        {'shape': [5, 4], 'objects': ['Floor'], 'colors': ['NONE']},
        # invalid (missing fields)
        {'objects': ['Floor'], 'colors': ['NONE']},
        {'shape': [5, 5], 'colors': ['NONE']},
        {'shape': [5, 5], 'objects': ['Floor']},
    ],
)
def test_factory_observation_space_fail(data):
    with pytest.raises(SchemaError):
        yaml_factory.factory_observation_space(data)


# NOTE: individual factory_xxx_function are annoying to test comprehensively;
# they are tested indirectly by testing the entire format in the following test


# NOTE testing all yaml files in yaml/
@pytest.mark.parametrize('path', glob.glob('yaml/*.yaml'))
def test_factory_rnv_from_yaml(path: str):
    env = yaml_factory.factory_env_from_yaml(path)
    assert isinstance(env, InnerEnv)
