from gym_gridverse.actions import Actions
from gym_gridverse.state import State


def static_reward(
    state: State,
    action: Actions,  # pylint: disable=unused-argument
    next_state: State,
) -> float:
    """negative reward if state is unchanged"""
    return -1.0 if state == next_state else 0.0
