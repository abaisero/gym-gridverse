import pytest

from gym_gridverse.envs.factory_yaml import make_environment


@pytest.mark.parametrize(
    'filename',
    [
        'data.yaml',
        'envs_yaml/env_minigrid_empty.yaml',
        'envs_yaml/env_minigrid_door_key.yaml',
        'envs_yaml/env_minigrid_dynamic_obstacles_16x16_random.v0.yaml',
    ],
)
def test_make_environment(filename: str):
    with open(filename) as f:
        make_environment(f)
