import pytest

from gym_gridverse.envs.factory import gym_minigrid_from_descr


def test_that_it_errors_for_coverage():
    with pytest.raises(ValueError):
        gym_minigrid_from_descr("blah!")
