from gym_gridverse.actions import Actions
from gym_gridverse.geometry import Orientation
from gym_gridverse.state import State


def actual_rotation_reward(
    state: State,
    action: Actions,  # pylint: disable=unused-argument
    next_state: State,
    *,
    reward_clockwise: float,
    reward_counterclockwise: float,
) -> float:
    """determines reward depending on whether agent has turned (counter)clockwise"""

    # we represent a rotation as the tuple of orientations before and after the
    # transition
    rotation = (state.agent.orientation, next_state.agent.orientation)

    # these are all the clockwise rotations
    clockwise_rotations = [
        (Orientation.N, Orientation.E),
        (Orientation.E, Orientation.S),
        (Orientation.S, Orientation.W),
        (Orientation.W, Orientation.N),
    ]

    if rotation in clockwise_rotations:
        return reward_clockwise

    # these are all the counterclockwise rotations
    counterclockwise_rotations = [
        (Orientation.N, Orientation.W),
        (Orientation.W, Orientation.S),
        (Orientation.S, Orientation.E),
        (Orientation.E, Orientation.N),
    ]

    if rotation in counterclockwise_rotations:
        return reward_counterclockwise

    return 0.0
