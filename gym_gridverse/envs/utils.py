from gym_gridverse.actions import Actions, TRANSLATION_ACTIONS
from gym_gridverse.geometry import Orientation, Position


def updated_agent_position_if_unobstructed(
    agent_pos: Position, agent_orientation: Orientation, action: Actions
) -> Position:
    """Returns the desired/intended position according to `action`

    NOTE: Assumes action is successful and next position is not blocking agent

    TODO: test

    Args:
        agent_pos (`Position`): current agent position
        action (`Actions`): action taken by agent

    Returns:
        Position: next position
    """
    if action not in TRANSLATION_ACTIONS:
        return agent_pos

    # Map directions to relative orientation
    direction_to_relative_orientation = {
        Actions.MOVE_FORWARD: agent_orientation,
        Actions.MOVE_LEFT: agent_orientation.rotate_left(),
        Actions.MOVE_RIGHT: agent_orientation.rotate_right(),
        Actions.MOVE_BACKWARD: agent_orientation.rotate_right().rotate_right(),
    }

    delta = direction_to_relative_orientation[action].as_delta_position()
    return Position.add(agent_pos, delta)
