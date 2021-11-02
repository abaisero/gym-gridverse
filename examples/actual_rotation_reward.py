from gym_gridverse.action import Action
from gym_gridverse.envs.reward_functions import reward_function_registry
from gym_gridverse.geometry import Orientation
from gym_gridverse.state import State

# these are all the clockwise rotations (represented as tuples)
_clockwise_rotations = {
    (Orientation.F, Orientation.R),
    (Orientation.R, Orientation.B),
    (Orientation.B, Orientation.L),
    (Orientation.L, Orientation.F),
}

# these are all the counterclockwise rotations (represented as tuples)
_counterclockwise_rotations = {
    (Orientation.F, Orientation.L),
    (Orientation.L, Orientation.B),
    (Orientation.B, Orientation.R),
    (Orientation.R, Orientation.F),
}


@reward_function_registry.register
def actual_rotation(
    state: State,
    action: Action,
    next_state: State,
    *,
    reward_clockwise: float,
    reward_counterclockwise: float,
) -> float:
    """determines reward depending on whether agent has turned (counter)clockwise"""

    # actual rotation represented as tuple of orientations in previous and next states
    rotation = (state.agent.orientation, next_state.agent.orientation)

    return (
        reward_clockwise
        if rotation in _clockwise_rotations
        else reward_counterclockwise
        if rotation in _counterclockwise_rotations
        else 0.0
    )
