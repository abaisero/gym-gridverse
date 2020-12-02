"""Defines the Action class"""

import enum


class Actions(enum.Enum):
    """An action is a glorified integer.

    In particular, there are 8 actions, each is assigned an integer:

    - 4 translations (forward, backwards, left & right)
    - 2 rotations (left & right)
    - ``actuate``
    - ``pick and drop``

    The effects of ``actuate`` depends on the object in front of the agent, but could
    open doors for example. ``pick and drop`` attempts to pick up the object.
    """

    MOVE_FORWARD = 0
    MOVE_BACKWARD = enum.auto()
    MOVE_LEFT = enum.auto()
    MOVE_RIGHT = enum.auto()

    TURN_LEFT = enum.auto()
    TURN_RIGHT = enum.auto()

    ACTUATE = enum.auto()
    PICK_N_DROP = enum.auto()


TRANSLATION_ACTIONS = [
    Actions.MOVE_FORWARD,
    Actions.MOVE_BACKWARD,
    Actions.MOVE_LEFT,
    Actions.MOVE_RIGHT,
]
"""A list of all the actions that moves the agent"""

ROTATION_ACTIONS = [Actions.TURN_LEFT, Actions.TURN_RIGHT]
"""A list of all the actions that rotates the agent"""
