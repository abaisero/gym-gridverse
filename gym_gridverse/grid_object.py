""" Every cell in the grid is represented by a grid objects """
from __future__ import annotations

import abc
import enum
from typing import TYPE_CHECKING, Callable, List, Optional, Type

import numpy.random as rnd

from gym_gridverse.geometry import get_manhattan_boundary
from gym_gridverse.rng import get_gv_rng_if_none

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

    @classmethod
    @abc.abstractmethod
    def can_be_represented_in_state(cls) -> bool:
        """ Returns whether the state_index fully represents the object state """

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
    def step(
        self,
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        """ Optional behavior of the object """

    @abc.abstractmethod
    def actuate(
        self, state: State, *, rng: Optional[rnd.Generator] = None
    ) -> None:
        """ The (optional) behavior upon actuation """

    @abc.abstractmethod
    def render_as_char(self) -> str:
        """ A single char representation of the object"""

    def __eq__(self, other) -> bool:
        if not isinstance(other, GridObject):
            return NotImplemented

        return (
            self.type_index == other.type_index
            and self.state_index == other.state_index
            and self.color == other.color
        )


class NoneGridObject(GridObject):
    """ object representing the absence of an object """

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

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

    def step(
        self,
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
    ) -> None:
        assert RuntimeError('should never be called')

    def actuate(
        self,
        state: State,
        *,
        rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
    ) -> None:
        assert RuntimeError('should never be called')

    def render_as_char(self) -> str:
        return " "


class Hidden(GridObject):
    """ object representing an unobservable cell """

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return False

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

    def step(
        self,
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def actuate(
        self,
        state: State,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def render_as_char(self) -> str:
        return "."


class Floor(GridObject):
    """ Most basic object in the grid, represents empty cell """

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

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

    def step(
        self,
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def actuate(
        self,
        state: State,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def render_as_char(self) -> str:
        return " "


class Wall(GridObject):
    """ The (second) most basic object in the grid: blocking cell """

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

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

    def step(
        self,
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def actuate(
        self,
        state: State,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def render_as_char(self) -> str:
        return "#"


class Goal(GridObject):
    """ The (second) most basic object in the grid: blocking cell """

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

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

    def step(
        self,
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def actuate(
        self,
        state: State,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def render_as_char(self) -> str:
        return "G"


class Door(GridObject):
    """A door is a grid

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

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

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

    def step(
        self,
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def actuate(
        self,
        state: State,
        *,
        rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
    ) -> None:
        """Attempts to open door

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
            if (
                isinstance(state.agent.obj, Key)
                and state.agent.obj.color == self.color
            ):
                self._state = self.Status.OPEN

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
        }[self._state]


class Key(GridObject):
    """ A key opens a door with the same color """

    def __init__(self, c: Colors):
        """ Creates a key of color `c` """
        self._color = c

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

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

    def step(
        self,
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def actuate(
        self, state: State, *, rng: Optional[rnd.Generator] = None
    ) -> None:
        pass

    def render_as_char(self) -> str:
        return "K"


class MovingObstacle(GridObject):
    """An obstacle to be avoided that moves in the grid"""

    def __init__(self):
        """Moving obstacles have no special status or color"""

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

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

    def step(
        self,
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        """Moves randomly

        Moves only to cells containing _Floor_ objects, and will do so with
        random walk. In current implementation can only move 1 cell
        non-diagonally. If (and only if) no open cells are available will it
        stay put

        Args:
            state (`State`): current state
            action (`Actions`): action taken by agent (ignored)
        """
        rng = get_gv_rng_if_none(None)

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

        next_position = rng.choice(proposed_next_positions)
        state.grid.swap(cur_pos, next_position)

    def actuate(
        self, state: State, *, rng: Optional[rnd.Generator] = None
    ) -> None:
        pass

    def render_as_char(self) -> str:
        return "*"


class Box(GridObject):
    """A box which can be broken and may contain another object"""

    def __init__(self, obj: GridObject):
        """Boxes have no special status or color"""
        if isinstance(obj, (NoneGridObject, Hidden)):
            raise ValueError(
                'box cannot contain NoneGridObject or Hidden types'
            )

        self.obj = obj

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return False

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
        return True

    def step(
        self,
        state: State,
        action: Actions,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> None:
        pass

    def actuate(
        self,
        state: State,
        *,
        rng: Optional[rnd.Generator] = None,  # pylint: disable=unused-argument
    ) -> None:

        position = state.grid.get_position(self)
        state.grid[position] = self.obj

    def render_as_char(self) -> str:
        return "b"


def factory(
    name: str,
    *,
    status: Optional[str] = None,
    color: Optional[str] = None,
    obj: Optional[GridObject] = None,
) -> GridObject:

    if name in ['none_grid_object', 'NoneGridObject']:
        return NoneGridObject()

    if name in ['hidden', 'Hidden']:
        return Hidden()

    if name in ['floor', 'Floor']:
        return Floor()

    if name in ['wall', 'Wall']:
        return Wall()

    if name in ['goal', 'Goal']:
        return Goal()

    if name in ['door', 'Door']:
        if not (isinstance(status, str) and isinstance(color, str)):
            raise ValueError(f'invalid parameters for name `{name}`')

        status_ = Door.Status[status]
        color_ = Colors[color]
        return Door(status_, color_)

    if name in ['key', 'Key']:
        if not isinstance(color, str):
            raise ValueError(f'invalid parameters for name `{name}`')

        color_ = Colors[color]
        return Key(color_)

    if name in ['moving_obstacle', 'MovingObstacle']:
        return MovingObstacle()

    if name in ['box', 'Box']:
        if not isinstance(obj, GridObject):
            raise ValueError(f'invalid parameters for name `{name}`')

        return Box(obj)

    raise ValueError(f'invalid grid-object name {name}')


def factory_type(name: str) -> Type[GridObject]:

    if name in ['none_grid_object', 'NoneGridObject']:
        return NoneGridObject

    if name in ['hidden', 'Hidden']:
        return Hidden

    if name in ['floor', 'Floor']:
        return Floor

    if name in ['wall', 'Wall']:
        return Wall

    if name in ['goal', 'Goal']:
        return Goal

    if name in ['door', 'Door']:
        return Door

    if name in ['key', 'Key']:
        return Key

    if name in ['moving_obstacle', 'MovingObstacle']:
        return MovingObstacle

    if name in ['box', 'Box']:
        return Box

    raise ValueError(f'invalid grid-object type name {name}')


GridObjectFactory = Callable[[], GridObject]
