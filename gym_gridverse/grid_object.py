""" Every cell in the grid is represented by a grid objects """
from __future__ import annotations

import abc
import enum
from typing import Callable, List, Optional, Type

from gym_gridverse.debugging import checkraise


class Color(enum.Enum):
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
    def color(self) -> Color:
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

    def __hash__(self):
        return hash((self.type_index, self.state_index, self.color))


class NoneGridObject(GridObject):
    """ object representing the absence of an object """

    type_index: int

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Color:
        return Color.NONE

    @classmethod
    def num_states(cls) -> int:
        return 1

    @property
    def transparent(self) -> bool:  # type: ignore
        assert False

    @property
    def can_be_picked_up(self) -> bool:  # type: ignore
        assert False

    @property
    def blocks(self) -> bool:  # type: ignore
        assert False

    def render_as_char(self) -> str:
        return " "

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Hidden(GridObject):
    """ object representing an unobservable cell """

    type_index: int

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return False

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Color:
        return Color.NONE

    @classmethod
    def num_states(cls) -> int:
        return 1

    @property
    def transparent(self) -> bool:
        return False

    @property
    def can_be_picked_up(self) -> bool:  # type: ignore
        assert False

    @property
    def blocks(self) -> bool:  # type: ignore
        assert False

    def render_as_char(self) -> str:
        return "."

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Floor(GridObject):
    """ Most basic object in the grid, represents empty cell """

    type_index: int

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Color:
        return Color.NONE

    @classmethod
    def num_states(cls) -> int:
        return 1

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return False

    def render_as_char(self) -> str:
        return " "

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Wall(GridObject):
    """ The (second) most basic object in the grid: blocking cell """

    type_index: int

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Color:
        return Color.NONE

    @classmethod
    def num_states(cls) -> int:
        return 1

    @property
    def transparent(self) -> bool:
        return False

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return True

    def render_as_char(self) -> str:
        return "#"

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Exit(GridObject):
    """ The (second) most basic object in the grid: blocking cell """

    type_index: int

    def __init__(self, color: Color = Color.NONE):
        self._color = color

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Color:
        return self._color

    @classmethod
    def num_states(cls) -> int:
        return 1

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return False

    def render_as_char(self) -> str:
        return "E"

    def __repr__(self):
        return f'{self.__class__.__name__}()'


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

    type_index: int

    class Status(enum.Enum):
        """ open, closed or locked """

        OPEN = 0
        CLOSED = enum.auto()
        LOCKED = enum.auto()

    def __init__(self, state: Door.Status, color: Color):
        self._color = color
        self._state = state

    @property
    def state(self) -> Door.Status:
        return self._state

    @state.setter
    def state(self, value: Door.Status):
        if not isinstance(value, Door.Status):
            return TypeError('value ({value}) must be of type Door.Status')

        self._state = value

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
    def color(self) -> Color:
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

    def __repr__(self):
        return f'{self.__class__.__name__}({self.state!s}, {self.color!s})'


class Key(GridObject):
    """ A key opens a door with the same color """

    type_index: int

    def __init__(self, c: Color):
        """ Creates a key of color `c` """
        self._color = c

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Color:
        return self._color

    @classmethod
    def num_states(cls) -> int:
        return 1

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return True

    @property
    def blocks(self) -> bool:
        return False

    def render_as_char(self) -> str:
        return "K"

    def __repr__(self):
        return f'{self.__class__.__name__}({self.color!s})'


class MovingObstacle(GridObject):
    """An obstacle to be avoided that moves in the grid"""

    type_index: int

    def __init__(self):
        """Moving obstacles have no special status or color"""

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Color:
        return Color.NONE

    @classmethod
    def num_states(cls) -> int:
        return 1

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return False

    def render_as_char(self) -> str:
        return "*"

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Box(GridObject):
    """A box which can be broken and may contain another object"""

    type_index: int

    def __init__(self, content: GridObject):
        """Boxes have no special status or color"""
        checkraise(
            lambda: not isinstance(content, (NoneGridObject, Hidden)),
            ValueError,
            'box cannot contain NoneGridObject or Hidden types',
        )

        self.content = content

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return False

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Color:
        return Color.NONE

    @classmethod
    def num_states(cls) -> int:
        return 1

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return True

    def render_as_char(self) -> str:
        return "b"

    def __repr__(self):
        return f'{self.__class__.__name__}({self.content!r})'


class Telepod(GridObject):
    """A teleportation pod"""

    type_index: int

    def __init__(self, color: Color):
        self._color = color

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Color:
        return self._color

    @classmethod
    def num_states(cls) -> int:
        return 1

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return False

    def render_as_char(self) -> str:
        return "T"

    def __repr__(self):
        return f'{self.__class__.__name__}({self.color!s})'


class Beacon(GridObject):
    """A beacon to attract attention"""

    type_index: int

    def __init__(self, color: Color):
        self._color = color

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @property
    def state_index(self) -> int:
        return 0

    @property
    def color(self) -> Color:
        return self._color

    @classmethod
    def num_states(cls) -> int:
        return 1

    @property
    def transparent(self) -> bool:
        return True

    @property
    def can_be_picked_up(self) -> bool:
        return False

    @property
    def blocks(self) -> bool:
        return False

    def render_as_char(self) -> str:
        return "B"

    def __repr__(self):
        return f'{self.__class__.__name__}({self.color!s})'


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

    if name in ['exit', 'Exit']:
        return Exit()

    if name in ['door', 'Door']:
        checkraise(
            lambda: isinstance(status, str) and isinstance(color, str),
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        status_ = Door.Status[status]
        color_ = Color[color]
        return Door(status_, color_)

    if name in ['key', 'Key']:
        checkraise(
            lambda: isinstance(color, str),
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        color_ = Color[color]
        return Key(color_)

    if name in ['moving_obstacle', 'MovingObstacle']:
        return MovingObstacle()

    if name in ['box', 'Box']:
        checkraise(
            lambda: isinstance(obj, GridObject),
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        return Box(obj)

    if name in ['telepod', 'Telepod']:
        checkraise(
            lambda: isinstance(color, str),
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        color_ = Color[color]
        return Telepod(color_)

    if name in ['beacon', 'Beacon']:
        checkraise(
            lambda: isinstance(color, str),
            ValueError,
            'invalid parameters for name `{}`',
            name,
        )

        color_ = Color[color]
        return Beacon(color_)

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

    if name in ['exit', 'Exit']:
        return Exit

    if name in ['door', 'Door']:
        return Door

    if name in ['key', 'Key']:
        return Key

    if name in ['moving_obstacle', 'MovingObstacle']:
        return MovingObstacle

    if name in ['box', 'Box']:
        return Box

    if name in ['telepod', 'Telepod']:
        return Telepod

    if name in ['beacon', 'Beacon']:
        return Beacon

    raise ValueError(f'invalid grid-object type name {name}')


GridObjectFactory = Callable[[], GridObject]
"""Signature for a function that instantiates grid objects on call"""
