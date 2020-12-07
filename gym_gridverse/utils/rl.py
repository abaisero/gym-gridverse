from typing import Callable


def make_return_computer(discount: float) -> Callable[[float], float]:
    """Coroutine which receives rewards and yields cumulative discounted returns"""

    cumreturn, cumdiscount = 0.0, 1.0

    def return_computer(reward: float) -> float:
        nonlocal cumreturn, cumdiscount
        cumreturn += cumdiscount * reward
        cumdiscount *= discount
        return cumreturn

    return return_computer
