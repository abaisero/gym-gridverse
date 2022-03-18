from __future__ import annotations

import enum

from gym_gridverse.grid_object import Color, GridObject


class Ice(GridObject):
    """An icy tile which can be `smooth` or `broken`."""

    color = Color.NONE
    blocks_vision = False
    holdable = False

    class Status(enum.Enum):
        SMOOTH = 0
        BROKEN = 0

    def __init__(self, state: Ice.Status):
        self.state = state

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return len(Ice.Status)

    @property
    def state_index(self) -> int:
        return self.state.value

    @property
    def blocks_movement(self) -> bool:
        return self.is_smooth

    @property
    def is_smooth(self) -> bool:
        return self.state is Ice.Status.SMOOTH

    @property
    def is_broken(self) -> bool:
        return self.state is Ice.Status.BROKEN

    def __repr__(self):
        return f'{self.__class__.__name__}({self.state!s})'
