""" Every cell in the grid is represented by a grid objects """
from __future__ import annotations

import abc
import enum
from collections import UserList
from typing import Callable, List, Type

from typing_extensions import TypeAlias


class Color(enum.Enum):
    """Color of grid objects"""

    NONE = 0
    RED = enum.auto()
    GREEN = enum.auto()
    BLUE = enum.auto()
    YELLOW = enum.auto()


class GridObjectRegistry(UserList):
    def register(self, object_type: Type[GridObject]) -> Type[GridObject]:
        self.data.append(object_type)
        return object_type

    def names(self) -> List[str]:
        """Returns the names of registered grid-objects"""
        return [object_type.__name__ for object_type in self.data]

    def from_name(self, name: str) -> Type[GridObject]:
        """Returns the grid-object class associated with a name"""
        try:
            return next(
                object_type
                for object_type in self.data
                if object_type.__name__ == name
            )
        except StopIteration as error:
            raise ValueError(f'Unregistered GridObject `{name}`') from error


grid_object_registry = GridObjectRegistry()
"""GridObject registry"""


class GridObjectMeta(abc.ABCMeta):
    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        # checks attribute existence at object instantiation
        obj.state_index
        obj.color
        obj.blocks_movement
        obj.blocks_vision
        obj.holdable
        return obj


class GridObject(metaclass=GridObjectMeta):
    """Represents the contents of a grid cell"""

    @property
    @abc.abstractmethod
    def state_index(self) -> int:
        """State index of this grid-object"""

    @property
    @abc.abstractmethod
    def color(self) -> Color:
        """Color of this grid-object"""

    @property
    @abc.abstractmethod
    def blocks_movement(self) -> bool:
        """Whether this grid-object blocks the agent from moving on it"""

    @property
    @abc.abstractmethod
    def blocks_vision(self) -> bool:
        """Whether this grid-object blocks the agent's vision."""

    @property
    @abc.abstractmethod
    def holdable(self) -> bool:
        """Whether the agent can pick up this grid-object"""

    def __init_subclass__(cls, *, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register:
            grid_object_registry.register(cls)

    @classmethod
    def type_index(cls) -> int:
        return grid_object_registry.index(cls)

    @classmethod
    @abc.abstractmethod
    def can_be_represented_in_state(cls) -> bool:
        """True iff :py:attr:`~gym_gridverse.grid_object.GridObject.state_index` fully represents the grid-object state.

        GridObjects may have an internal state which is not fully representable
        by a single integer
        :py:attr:`~gym_gridverse.grid_object.GridObject.state_index`, e.g., a
        :py:class:`~gym_gridverse.grid_object.Box` contains a reference to
        another :py:class:`~gym_gridverse.grid_object.GridObject` as its
        content.  The unfortunate implication is that this
        :py:class:`~gym_gridverse.grid_object.GridObject` (and, by extension,
        any Grido or Environment which contains this type of
        :py:class:`~gym_gridverse.grid_object.GridObject`) cannot produce a
        truly fully observable State representation, which becomes disallowed.
        However, the GridObject, Grid, and Environment may still be used to
        represent partially observable control tasks."""
        assert False

    @classmethod
    @abc.abstractmethod
    def num_states(cls) -> int:
        """Number of internal states.

        GridObjects themselves can have internal states, e.g., a Door may be
        ``open``, ``closed``, or ``locked``.  This classmethod return the
        number of possible states that this GridObject may have.
        """
        assert False

    def __eq__(self, other) -> bool:
        if not isinstance(other, GridObject):
            return NotImplemented

        return (
            self.type_index() == other.type_index()
            and self.state_index == other.state_index
            and self.color == other.color
        )

    def __hash__(self):
        return hash((self.type_index(), self.state_index, self.color))


class NoneGridObject(GridObject):
    """An object which represents the complete absence of any other object."""

    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = True
    holdable = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Hidden(GridObject):
    """An object which represents some other unobservable object."""

    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = True
    holdable = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return False

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Floor(GridObject):
    """An empty walkable spot"""

    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = False
    holdable = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Wall(GridObject):
    """An object which obstructs movement and vision."""

    state_index = 0
    color = Color.NONE
    blocks_movement = True
    blocks_vision = True
    holdable = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Exit(GridObject):
    """The (second) most basic object in the grid: blocking cell"""

    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = False
    holdable = False

    def __init__(self, color: Color = Color.NONE):
        super().__init__()
        self.color = color

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Door(GridObject):
    """A door which can be `open`, `closed` or `locked`.

    The following dynamics (upon actuation) occur:

    When not holding correct key with correct color:
        `open` or `closed` -> `open`
        `locked` -> `locked`

    When holding correct key:
        any state -> `open`

    Can be `OPEN`, `CLOSED` or `LOCKED`.
    """

    color = Color.NONE
    holdable = False

    state: Status

    class Status(enum.Enum):
        OPEN = 0
        CLOSED = enum.auto()
        LOCKED = enum.auto()

    def __init__(self, state: Door.Status, color: Color):
        super().__init__()
        self.state = state
        self.color = color

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return len(Door.Status)

    @property
    def state_index(self) -> int:
        return self.state.value

    @property
    def blocks_movement(self) -> bool:
        return not self.is_open

    @property
    def blocks_vision(self) -> bool:
        return not self.is_open

    @property
    def is_open(self) -> bool:
        """returns whether the door is opened."""
        return self.state is Door.Status.OPEN

    @property
    def is_locked(self) -> bool:
        """returns whether the door is locked."""
        return self.state is Door.Status.LOCKED

    def __repr__(self):
        return f'{self.__class__.__name__}({self.state!s}, {self.color!s})'


class Key(GridObject):
    """A key to open locked doors."""

    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = False
    holdable = True

    def __init__(self, color: Color):
        super().__init__()
        self.color = color

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}({self.color!s})'


class MovingObstacle(GridObject):
    """An obstacle to be avoided that moves in the grid."""

    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = False
    holdable = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Box(GridObject):
    """A box which can be broken and may contain another object."""

    state_index = 0
    color = Color.NONE
    blocks_movement = True
    blocks_vision = False
    holdable = False

    def __init__(self, content: GridObject):
        super().__init__()

        """Boxes have no special status or color"""
        if isinstance(content, (NoneGridObject, Hidden)):
            raise ValueError('Box cannot contain NoneGridObject or Hidden')

        self.content = content

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return False

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}({self.content!r})'


class Telepod(GridObject):
    """A pod which teleports elsewhere."""

    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = False
    holdable = False

    def __init__(self, color: Color):
        super().__init__()
        self.color = color

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}({self.color!s})'


class Beacon(GridObject):
    """A object to attract attention or convey information."""

    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = False
    holdable = False

    def __init__(self, color: Color):
        self.color = color

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}({self.color!s})'


GridObjectFactory: TypeAlias = Callable[[], GridObject]
"""A callable which returns grid objects."""
