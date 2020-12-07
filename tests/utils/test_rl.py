from typing import Iterable

import pytest

from gym_gridverse.utils.rl import make_return_computer


@pytest.mark.parametrize(
    'discount,rewards,expected',
    [
        (1.0, [1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]),
        (1.0, [1.0, -1.0, 1.0, -1.0], [1.0, 0.0, 1.0, 0.0]),
        (0.1, [1.0, 1.0, 1.0, 1.0], [1.0, 1.1, 1.11, 1.111]),
        (0.1, [1.0, -1.0, 1.0, -1.0], [1.0, 0.9, 0.91, 0.909]),
    ],
)
def test_make_return_computer(
    discount: float, rewards: Iterable[float], expected: Iterable[float]
):
    return_computer = make_return_computer(discount)

    for reward, ret in zip(rewards, expected):
        assert return_computer(reward) == ret
