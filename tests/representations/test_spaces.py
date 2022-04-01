import numpy as np
import pytest

from gym_gridverse.representations.spaces import Space, SpaceType

# CATEGORICAL


@pytest.mark.parametrize(
    'upper_bound',
    [
        np.array([0, 1, 2]),
        np.array([1, 2, 3]),
        np.array([2, 3, 4]),
    ],
)
def test_categorical_space(upper_bound: np.ndarray):
    space = Space.make_categorical_space(upper_bound)
    assert space.space_type is SpaceType.CATEGORICAL
    assert np.all(space.lower_bound == np.zeros_like(upper_bound))
    assert np.all(space.upper_bound == upper_bound)


@pytest.mark.parametrize(
    'upper_bound',
    [
        # wrong values
        np.array([0, 0, -1]),
        np.array([0, -1, 0]),
        np.array([-1, 0, 0]),
        # wrong dtype
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0, 4.0]),
    ],
)
def test_categorical_space_fail(upper_bound: np.ndarray):
    with pytest.raises(ValueError):
        Space.make_categorical_space(upper_bound)


@pytest.mark.parametrize(
    'upper_bound,x,expected',
    [
        (np.array([1, 2, 3]), np.array([0, 0, 0]), True),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), True),
        (np.array([1, 2, 3]), np.array([-1, -1, -1]), False),
        (np.array([1, 2, 3]), np.array([2, 3, 4]), False),
    ],
)
def test_categorical_space_contains(
    upper_bound: np.ndarray, x: np.ndarray, expected: bool
):
    space = Space.make_categorical_space(upper_bound)
    assert space.contains(x) == expected


# DISCRETE


@pytest.mark.parametrize(
    'lower_bound,upper_bound',
    [
        (np.array([0, 1, 2]), np.array([1, 2, 3])),
        (np.array([1, 2, 3]), np.array([2, 3, 4])),
        (np.array([2, 3, 4]), np.array([3, 4, 5])),
    ],
)
def test_discrete_space(lower_bound: np.ndarray, upper_bound: np.ndarray):
    space = Space.make_discrete_space(lower_bound, upper_bound)
    assert space.space_type is SpaceType.DISCRETE
    assert np.all(space.lower_bound == lower_bound)
    assert np.all(space.upper_bound == upper_bound)


@pytest.mark.parametrize(
    'lower_bound,upper_bound',
    [
        # wrong values
        (np.array([0, 0, 1]), np.array([0, 0, -1])),
        (np.array([0, 1, 0]), np.array([0, -1, 0])),
        (np.array([1, 0, 0]), np.array([-1, 0, 0])),
        # wrong shapes
        (np.array([0, 1, 2]), np.array([1])),
        (np.array([1, 2, 3]), np.array([1, 2])),
        (np.array([2, 3, 4]), np.array([1, 2, 3, 4])),
        # wrong dtype
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])),
        (np.array([2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0])),
    ],
)
def test_discrete_space_fail(lower_bound: np.ndarray, upper_bound: np.ndarray):
    with pytest.raises(ValueError):
        Space.make_discrete_space(lower_bound, upper_bound)


@pytest.mark.parametrize(
    'lower_bound,upper_bound,x,expected',
    [
        (np.array([1, 2]), np.array([3, 4]), np.array([0, 1]), False),
        (np.array([1, 2]), np.array([3, 4]), np.array([1, 2]), True),
        (np.array([1, 2]), np.array([3, 4]), np.array([2, 3]), True),
        (np.array([1, 2]), np.array([3, 4]), np.array([3, 4]), True),
        (np.array([1, 2]), np.array([3, 4]), np.array([4, 5]), False),
    ],
)
def test_discrete_space_contains(
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    x: np.ndarray,
    expected: bool,
):
    space = Space.make_discrete_space(lower_bound, upper_bound)
    assert space.contains(x) == expected


# CONTINUOUS


@pytest.mark.parametrize(
    'lower_bound,upper_bound',
    [
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])),
        (np.array([2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0])),
    ],
)
def test_continuous_space(lower_bound: np.ndarray, upper_bound: np.ndarray):
    space = Space.make_continuous_space(lower_bound, upper_bound)
    assert space.space_type is SpaceType.CONTINUOUS
    assert np.all(space.lower_bound == lower_bound)
    assert np.all(space.upper_bound == upper_bound)


@pytest.mark.parametrize(
    'lower_bound,upper_bound',
    [
        # wrong values
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])),
        (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
        # wrong shapes
        (np.array([0.0, 1.0, 2.0]), np.array([1.0])),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])),
        (np.array([2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0])),
        # wrong dtype
        (np.array([0, 1, 2]), np.array([1, 2, 3])),
        (np.array([1, 2, 3]), np.array([2, 3, 4])),
        (np.array([2, 3, 4]), np.array([3, 4, 5])),
    ],
)
def test_continuous_space_fail(
    lower_bound: np.ndarray, upper_bound: np.ndarray
):
    with pytest.raises(ValueError):
        Space.make_continuous_space(lower_bound, upper_bound)


@pytest.mark.parametrize(
    'lower_bound,upper_bound,x,expected',
    [
        (
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([0.0, 1.0]),
            False,
        ),
        (
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([1.0, 2.0]),
            True,
        ),
        (
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([2.0, 3.0]),
            True,
        ),
        (
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([3.0, 4.0]),
            True,
        ),
        (
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([4.0, 5.0]),
            False,
        ),
    ],
)
def test_continuous_space_contains(
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    x: np.ndarray,
    expected: bool,
):
    space = Space.make_continuous_space(lower_bound, upper_bound)
    assert space.contains(x) == expected
