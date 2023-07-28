import glob

import pytest

import yaml
from gym_gridverse.envs.yaml.schemas import schemas


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
    assert schemas['shape'].is_valid(data) == expected


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
    assert schemas['layout'].is_valid(data) == expected


@pytest.mark.parametrize(
    'data,expected',
    [
        ({'objects': ['Floor'], 'colors': ['NONE']}, True),
        # invalid (missing fields)
        ({'colors': ['NONE']}, False),
        ({'objects': ['Floor']}, False),
    ],
)
def test_state_space_schema(data, expected: bool):
    assert schemas['state_space'].is_valid(data) == expected


@pytest.mark.parametrize(
    'data,expected',
    [
        (['MOVE_FORWARD', 'MOVE_BACKWARD'], True),
        (['MOVE_FORWARD'], True),
        # invalid (wrong length)
        ([], False),
        # invalid (contains duplicates)
        (['MOVE_FORWARD', 'MOVE_FORWARD'], False),
    ],
)
def test_action_space_schema(data, expected: bool):
    assert schemas['action_space'].is_valid(data) == expected


@pytest.mark.parametrize(
    'data,expected',
    [
        ({'objects': ['Floor'], 'colors': ['NONE']}, True),
        # invalid (missing fields)
        ({'colors': ['NONE']}, False),
        ({'objects': ['Floor']}, False),
    ],
)
def test_observation_space_schema(data, expected: bool):
    assert schemas['observation_space'].is_valid(data) == expected


# NOTE: individual xxx_function_schema are annoying to test comprehensively;
# they are tested indirectly by testing the entire format in the following test


# NOTE testing all yaml files in yaml/
@pytest.mark.parametrize('path', glob.glob('yaml/*.yaml'))
def test_env_schema(path: str):
    with open(path) as f:
        data = yaml.safe_load(f)

    assert schemas['env'].is_valid(data)
