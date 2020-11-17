import pytest

from gym_gridverse.rng import (
    get_gv_rng,
    get_gv_rng_if_none,
    make_rng,
    reset_gv_rng,
)


def test_make_rng():
    rng1 = make_rng()
    rng2 = make_rng()
    assert rng1 is not rng2


def test_reset_gv_rng_wo_seed():
    num_samples = 10

    rng = reset_gv_rng()
    values1 = [rng.normal() for _ in range(num_samples)]

    rng = reset_gv_rng()
    values2 = [rng.normal() for _ in range(num_samples)]

    assert values1 != values2


@pytest.mark.parametrize('seed1', [1, 2, 1337, 0xDEADBEEF])
@pytest.mark.parametrize('seed2', [1, 2, 1337, 0xDEADBEEF])
def test_reset_gv_rng_w_seed(seed1: int, seed2: int):
    num_samples = 10

    # seed and get some values
    rng = reset_gv_rng(seed1)
    values1 = [rng.normal() for _ in range(num_samples)]

    # re-seed and get some other values
    rng = reset_gv_rng(seed2)
    values2 = [rng.normal() for _ in range(num_samples)]

    # values are the same only if seeds are the same
    assert (values1 == values2) == (seed1 == seed2)


def test_get_gv_rng():
    # two calls return the same rng
    rng1 = get_gv_rng()
    rng2 = get_gv_rng()
    assert rng1 is rng2

    # two calls separated by a reset return different rngs
    rng1 = get_gv_rng()
    reset_gv_rng()
    rng2 = get_gv_rng()
    assert rng1 is not rng2


def test_rng_if_none():
    # default call returns library rng
    rng = None
    assert get_gv_rng_if_none(rng) is get_gv_rng()

    # call with an rng returns that rng
    rng = make_rng()
    assert get_gv_rng_if_none(rng) is rng
