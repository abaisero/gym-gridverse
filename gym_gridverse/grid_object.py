""" Every cell in the grid is represented by a grid objects """
from __future__ import annotations

import abc
import enum
from typing import List

import numpy as np


class Colors(enum.Enum):
    """ Color of grid objects """

    NONE = 0
    RED = enum.auto()
    GREEN = enum.auto()
    BLUE = enum.auto()
    YELLOW = enum.auto()


class GridObject(metaclass=abc.ABCMeta):
    """ A cell in the grid """

    # registry as list/mapping int -> GridObject
    object_types: List[GridObject] = []
    type_index: int

    def __init_subclass__(cls, *, noregister=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not noregister:
            cls.type_index = len(GridObject.object_types)
            GridObject.object_types.append(cls)

    @property
    @abc.abstractmethod
    def state_index(self) -> int:
        """ Returns the state index of the object """

    @property
    @abc.abstractmethod
    def color(self) -> Colors:
        """ returns the color of the object """

    @property
    @abc.abstractmethod
    def transparent(self) -> bool:
        """ Whether the agent can see _through_ the object """

    @property
    @abc.abstractmethod
    def can_be_picked_up(self) -> bool:
        """ Whether the agent can see pick up the object """

    @property
    @abc.abstractmethod
    def blocks(self) -> bool:
        """ Whether the object blocks the agent """

    @abc.abstractmethod
    def step(self, state) -> None:  # TODO: state annotate?
        """ Optional behavior of the object """

    @abc.abstractmethod
    def actuate(self, state) -> None:  # TODO: state annotate?
        """ The (optional) behavior upon actuation """

    def as_array(self):
        """ A (3,) array representation of the object """
        return np.array([self.type_index, self.state_index, self.color.value])


class Floor(GridObject):
    """ Most basic object in the grid, represents empty cell """

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Colors:
        return Colors.NONE

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return False

    def step(self, state) -> None:
        pass

    def actuate(self, state) -> None:
        pass


class Wall(GridObject):
    """ The (second) most basic object in the grid: blocking cell """

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Colors:
        return Colors.NONE

    @property
    def transparent(self) -> bool:
        return False

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return True

    def step(self, state) -> None:
        pass

    def actuate(self, state) -> None:
        pass


class Goal(GridObject):
    """ The (second) most basic object in the grid: blocking cell """

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Colors:
        return Colors.NONE

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return False

    def step(self, state) -> None:
        pass

    def actuate(self, state) -> None:
        pass


class Door(GridObject):
    """ A door is a grid

    Can be `open`, `closed` or `locked`.

    The following dynamics (upon actuation) occur:

    When not holding correct key with correct color:
        `open` or `closed` -> `open`
        `locked` -> `locked`

    When holding correct key:
        any state -> `open`

    """

    class State(enum.Enum):
        """ open, closed or locked """

        OPEN = 0
        CLOSED = enum.auto()
        LOCKED = enum.auto()

    def __init__(self, state: State, color: Colors):
        self._color = color
        self._state = state

    @property
    def state_index(self) -> int:
        return self._state.value

    @property
    def color(self) -> Colors:
        return self._color

    @property
    def transparent(self) -> bool:
        return self._state == self.State.OPEN

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return not self._state == self.State.OPEN

    def step(self, state) -> None:
        pass

    def actuate(self, state) -> None:
        """ Attempts to open door 

        When not holding correct key with correct color:
            `open` or `closed` -> `open`
            `locked` -> `locked`

        When holding correct key:
            any state -> `open`

        """

        if self.is_open:
            return
        if not self.locked:
            self._state = self.State.OPEN
        else:
            try:
                # TODO: test through type index or isinstance?
                if (
                    isinstance(state.agent.obj, Key)
                    and state.agent.obj.color == self.color
                ):
                    self._state = self.State.OPEN
                # TODO: disappear key?
            except:
                pass

    @property
    def is_open(self) -> bool:
        """ returns whether the door is opened """
        return self._state == self.State.OPEN

    @property
    def locked(self) -> bool:
        """ returns whether the door is locked """
        return self._state == self.State.LOCKED


class Key(GridObject):
    """ A key opens a door with the same color """

    def __init__(self, c: Colors):
        """ Creates a key of color `c` """
        self._color = c

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Colors:
        return self._color

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return True

    @property
    def blocks(self) -> bool:
        return False

    def step(self, state) -> None:  # TODO: state annotate?
        pass

    def actuate(self, state) -> None:  # TODO: state annotate?
        pass
