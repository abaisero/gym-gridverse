import enum


class Actions(enum.Enum):
    """A glorified integer

    Actions have been grouped together, in particular there are
    `TRANSLATION_ACTIONS` and `ROTATION_ACTIONS` that are lists of actions that
    either 'move' or 'rotate'.
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

ROTATION_ACTIONS = [Actions.TURN_LEFT, Actions.TURN_RIGHT]
