import glob

import pytest

import yaml
from gym_gridverse.envs.yaml import schemas


@pytest.mark.parametrize(
    'data,expected',
    [
        ([1, 2], True),
        ([2, 1], True),
        # invalid (non-positive)
        ([0, 1], False),
        ([1, 0], False),
        # invalid (wrong length)
        ([], False),
        ([1], False),
        ([1, 1, 1], False),
    ],
)
def test_shape_schema(data, expected: bool):
    schema = schemas.shape_schema()
    assert schema.is_valid(data) == expected


@pytest.mark.parametrize(
    'data,expected',
    [
        ([1, 2], True),
        ([2, 1], True),
        # non-positive
        ([0, 1], False),
        ([1, 0], False),
        # wrong length
        ([], False),
        ([1], False),
        ([1, 1, 1], False),
    ],
)
def test_layout_schema(data, expected: bool):
    schema = schemas.layout_schema()
    assert schema.is_valid(data) == expected


@pytest.mark.parametrize(
    'data,expected',
    [
        ('floor', True),
        ('wall', True),
        ('exit', True),
        ('door', True),
        ('key', True),
        ('moving_obstacle', True),
        ('box', True),
        ('telepod', True),
        ('beacon', True),
        # TODO: remove these
        ('Floor', True),
        ('Wall', True),
        ('Exit', True),
        ('Door', True),
        ('Key', True),
        ('MovingObstacle', True),
        ('Box', True),
        ('Telepod', True),
        ('Beacon', True),
        ('invalid', False),
    ],
)
def test_object_type_schema(data, expected: bool):
    schema = schemas.object_type_schema()
    assert schema.is_valid(data) == expected


@pytest.mark.parametrize(
    'data,expected',
    [
        ({'shape': [11, 11], 'objects': ['Floor'], 'colors': ['NONE']}, True),
        # invalid (missing fields)
        ({'objects': ['Floor'], 'colors': ['NONE']}, False),
        ({'shape': [11, 11], 'colors': ['NONE']}, False),
        ({'shape': [11, 11], 'objects': ['Floor']}, False),
    ],
)
def test_state_space_schema(data, expected: bool):
    schema = schemas.state_space_schema()
    assert schema.is_valid(data) == expected


@pytest.mark.parametrize(
    'data,expected',
    [
        (['MOVE_FORWARD', 'MOVE_BACKWARD'], True),
        # invalid (wrong length)
        ([], False),
        (['MOVE_FORWARD'], False),
        # invalid (contains duplicates)
        (['MOVE_FORWARD', 'MOVE_FORWARD'], False),
    ],
)
def test_action_space_schema(data, expected: bool):
    schema = schemas.action_space_schema()
    assert schema.is_valid(data) == expected


@pytest.mark.parametrize(
    'data,expected',
    [
        ({'shape': [11, 11], 'objects': ['Floor'], 'colors': ['NONE']}, True),
        # invalid (even width)
        ({'shape': [11, 10], 'objects': ['Floor'], 'colors': ['NONE']}, False),
        # invalid (missing fields)
        ({'objects': ['Floor'], 'colors': ['NONE']}, False),
        ({'shape': [11, 11], 'colors': ['NONE']}, False),
        ({'shape': [11, 11], 'objects': ['Floor']}, False),
    ],
)
def test_observation_space_schema(data, expected: bool):
    schema = schemas.observation_space_schema()
    assert schema.is_valid(data) == expected


# NOTE: individual xxx_function_schema are annoying to test comprehensively;
# they are tested indirectly by testing the entire format in the following test

# NOTE testing all yaml files in yaml/
@pytest.mark.parametrize('path', glob.glob('yaml/*.yaml'))
def test_env_schema(path: str):
    with open(path) as f:
        data = yaml.safe_load(f)

    schema = schemas.env_schema()
    assert schema.is_valid(data)
