from gym_gridverse.action import TRANSLATION_ACTIONS, Action
from gym_gridverse.geometry import Orientation, Position, PositionOrTuple


def updated_agent_position_if_unobstructed(
    agent_pos: PositionOrTuple, agent_orientation: Orientation, action: Action
) -> Position:
    """Returns the desired/intended position according to `action`

    NOTE: Assumes action is successful and next position is not blocking agent

    Args:
        agent_pos (`Position`): current agent position
        action (`Action`): action taken by agent

    Returns:
        Position: next position
    """
    agent_pos = Position.from_position_or_tuple(agent_pos)

    if action not in TRANSLATION_ACTIONS:
        return agent_pos

    # Map directions to relative orientation
    direction_to_relative_orientation = {
        Action.MOVE_FORWARD: agent_orientation,
        Action.MOVE_LEFT: agent_orientation.rotate_left(),
        Action.MOVE_RIGHT: agent_orientation.rotate_right(),
        Action.MOVE_BACKWARD: agent_orientation.rotate_back(),
    }

    delta = direction_to_relative_orientation[action].as_position()
    return agent_pos + delta
