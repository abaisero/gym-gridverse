"""Defines the Action class"""

import enum


class Action(enum.Enum):
    """Actions available to the agent.

    There are (up to) 8 actions:

    * 4 ``movement`` actions (forward, backwards, left & right)
    * 2 ``turn`` actions (left & right)
    * 1 ``actuate`` action, which can actuate objects (e.g., the one in front)
    * 1 ``pick and drop`` action, to pick up objects (e.g., the one in front)
    """

    MOVE_FORWARD = 0
    MOVE_BACKWARD = enum.auto()
    MOVE_LEFT = enum.auto()
    MOVE_RIGHT = enum.auto()

    TURN_LEFT = enum.auto()
    TURN_RIGHT = enum.auto()

    ACTUATE = enum.auto()
    PICK_N_DROP = enum.auto()

    def is_move(self) -> bool:
        """True if the action is a ``movement`` action"""
        return self in _MOVE_ACTIONS

    def is_turn(self) -> bool:
        """True if the action is a ``turn`` action"""
        return self in _TURN_ACTIONS


_MOVE_ACTIONS = {
    Action.MOVE_FORWARD,
    Action.MOVE_BACKWARD,
    Action.MOVE_LEFT,
    Action.MOVE_RIGHT,
}

_TURN_ACTIONS = {Action.TURN_LEFT, Action.TURN_RIGHT}
