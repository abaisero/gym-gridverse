import glob
import itertools as itt

import pytest
from schema import SchemaError

import gym_gridverse.envs.yaml.factory as yaml_factory
from gym_gridverse.action import Action
from gym_gridverse.envs import InnerEnv
from gym_gridverse.geometry import Shape
from gym_gridverse.spaces import ActionSpace


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
        ['MOVE_FORWARD', 'MOVE_BACKWARD'],
    ],
)
def test_factory_action_space(data):
    action_space = yaml_factory.factory_action_space(data)
    assert isinstance(action_space, ActionSpace)
    assert len(set(action_space.actions)) == len(action_space.actions)

    for action in action_space.actions:
        isinstance(action, Action)


@pytest.mark.parametrize(
    'data',
    [
        [],
        ['MOVE_FORWARD', 'MOVE_FORWARD'],
    ],
)
def test_factory_action_space_fail(data):
    with pytest.raises(SchemaError):
        yaml_factory.factory_action_space(data)


# NOTE: individual factory_xxx_function are annoying to test comprehensively;
# they are tested indirectly by testing the entire format from the yaml files


# NOTE testing all yaml files in yaml/
@pytest.mark.parametrize('path', glob.glob('yaml/*.yaml'))
def test_factory_rnv_from_yaml(path: str):
    env = yaml_factory.factory_env_from_yaml(path)
    assert isinstance(env, InnerEnv)

    # testing for runtime errors
    env.reset()
    for action in itt.islice(itt.cycle(env.action_space.actions), 10):
        _, done = env.step(action)
        if done:
            env.reset()
