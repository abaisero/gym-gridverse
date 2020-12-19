import pytest

from gym_gridverse.envs.factory import env_from_descr


def test_that_it_errors_for_coverage():
    with pytest.raises(ValueError):
        env_from_descr("blah!")
