""" Every cell in the grid is represented by a grid objects """
from __future__ import annotations

import abc
import enum
import random
from typing import TYPE_CHECKING, List

import numpy as np

from gym_gridverse.geometry import get_manhattan_boundary

if TYPE_CHECKING:
    from gym_gridverse.actions import Actions
    from gym_gridverse.state import State


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

    def __init_subclass__(cls, *, register=True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register:
            cls.type_index = len(GridObject.object_types)
            GridObject.object_types.append(cls)

    @property
    @abc.abstractmethod
    def state_index(self) -> int:
        """ Returns the state index of the object """

    @classmethod
    @abc.abstractmethod
    def num_states(cls) -> int:
        """ Number of states this class can take on"""

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
    def step(self, state: State, action: Actions) -> None:
        """ Optional behavior of the object """

    @abc.abstractmethod
    def actuate(self, state: State) -> None:
        """ The (optional) behavior upon actuation """

    def as_array(self):
        """ A (3,) array representation of the object """
        return np.array([self.type_index, self.state_index, self.color.value])

    @abc.abstractmethod
    def render_as_char(self) -> str:
        """ A single char representation of the object"""

    def __eq__(self, other) -> bool:
        if not isinstance(other, GridObject):
            return NotImplemented

        # TODO test equality for various gridobjects
        return (self.as_array() == other.as_array()).all()


class NoneGridObject(GridObject):
    """ object representing the absence of an object """

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Colors:
        return Colors.NONE

    @classmethod
    def num_states(cls) -> int:
        return 0

    @property
    def transparent(self) -> bool:  # type: ignore
        assert RuntimeError('should never be called')

    @property
    def can_be_picked_up(self) -> bool:  # type: ignore
        assert RuntimeError('should never be called')

    @property
    def blocks(self) -> bool:  # type: ignore
        assert RuntimeError('should never be called')

    def step(self, state: State, action: Actions) -> None:
        assert RuntimeError('should never be called')

    def actuate(self, state: State) -> None:
        assert RuntimeError('should never be called')

    def render_as_char(self) -> str:
        return " "


class Hidden(GridObject):
    """ object representing an unobservable cell """

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Colors:
        return Colors.NONE

    @classmethod
    def num_states(cls) -> int:
        return 0

    @property
    def transparent(self) -> bool:
        return False

    @property
    def can_be_picked_up(self) -> bool:  # type: ignore
        assert RuntimeError('should never be called')

    @property
    def blocks(self) -> bool:  # type: ignore
        assert RuntimeError('should never be called')

    def step(self, state: State, action: Actions) -> None:
        pass

    def actuate(self, state: State) -> None:
        pass

    def render_as_char(self) -> str:
        return "."


class Floor(GridObject):
    """ Most basic object in the grid, represents empty cell """

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Colors:
        return Colors.NONE

    @classmethod
    def num_states(cls) -> int:
        return 0

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return False

    def step(self, state: State, action: Actions) -> None:
        pass

    def actuate(self, state: State) -> None:
        pass

    def render_as_char(self) -> str:
        return " "


class Wall(GridObject):
    """ The (second) most basic object in the grid: blocking cell """

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Colors:
        return Colors.NONE

    @classmethod
    def num_states(cls) -> int:
        return 0

    @property
    def transparent(self) -> bool:
        return False

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return True

    def step(self, state: State, action: Actions) -> None:
        pass

    def actuate(self, state: State) -> None:
        pass

    def render_as_char(self) -> str:
        return "#"


class Goal(GridObject):
    """ The (second) most basic object in the grid: blocking cell """

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Colors:
        return Colors.NONE

    @classmethod
    def num_states(cls) -> int:
        return 0

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return False

    def step(self, state: State, action: Actions) -> None:
        pass

    def actuate(self, state: State) -> None:
        pass

    def render_as_char(self) -> str:
        return "G"


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

    class Status(enum.Enum):
        """ open, closed or locked """

        OPEN = 0
        CLOSED = enum.auto()
        LOCKED = enum.auto()

    def __init__(self, state: Status, color: Colors):
        self._color = color
        self._state = state

    @property
    def state_index(self) -> int:
        return self._state.value

    @classmethod
    def num_states(cls) -> int:
        return len(Door.Status)

    @property
    def color(self) -> Colors:
        return self._color

    @property
    def transparent(self) -> bool:
        return self._state == self.Status.OPEN

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return self._state != self.Status.OPEN

    def step(self, state: State, action: Actions) -> None:
        pass

    def actuate(self, state: State) -> None:
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
            self._state = self.Status.OPEN
        else:
            try:
                if (
                    isinstance(state.agent.obj, Key)
                    and state.agent.obj.color == self.color
                ):
                    self._state = self.Status.OPEN
            except:
                pass  # door is locked but agent does not hold the correct

    @property
    def is_open(self) -> bool:
        """ returns whether the door is opened """
        return self._state == self.Status.OPEN

    @property
    def locked(self) -> bool:
        """ returns whether the door is locked """
        return self._state == self.Status.LOCKED

    def render_as_char(self) -> str:
        return {
            self.Status.OPEN: "_",
            self.Status.CLOSED: "d",
            self.Status.LOCKED: "D",
        }.get(self._state)


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

    @classmethod
    def num_states(cls) -> int:
        return 0

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return True

    @property
    def blocks(self) -> bool:
        return False

    def step(self, state: State, action: Actions) -> None:
        pass

    def actuate(self, state: State) -> None:
        pass

    def render_as_char(self) -> str:
        return "K"


class MovingObstacle(GridObject):
    """An obstacle to be avoided that moves in the grid"""

    def __init__(self):
        """Moving obstacles have no special status or color"""

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Colors:
        return Colors.NONE

    @classmethod
    def num_states(cls) -> int:
        return 0

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return False

    def step(self, state: State, action: Actions) -> None:
        """Moves randomly

        Moves only to cells containing _Floor_ objects, and will do so with
        random walk. In current implementation can only move 1 cell
        non-diagonally. If (and only if) no open cells are available will it
        stay put

        Args:
            state ([TODO:type]): current state
            action (Actions): action taken by agent (ignored)
        """

        cur_pos = state.grid.get_position(self)

        proposed_next_positions = get_manhattan_boundary(cur_pos, distance=1)

        # Filter on next position is Floor and in grid
        proposed_next_positions = [
            x
            for x in proposed_next_positions
            if x in state.grid and isinstance(state.grid[x], Floor)
        ]

        if not proposed_next_positions:
            return

        next_position = random.choice(proposed_next_positions)
        state.grid.swap(cur_pos, next_position)

    def actuate(self, state: State) -> None:
        pass

    def render_as_char(self) -> str:
        return "*"
