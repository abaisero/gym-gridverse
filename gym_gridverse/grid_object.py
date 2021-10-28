""" Every cell in the grid is represented by a grid objects """
from __future__ import annotations

import abc
import enum
from collections import UserList
from typing import Callable, List, Type


class Color(enum.Enum):
    """Color of grid objects"""

    NONE = 0
    RED = enum.auto()
    GREEN = enum.auto()
    BLUE = enum.auto()
    YELLOW = enum.auto()


class GridObjectRegistry(UserList):
    def register(self, object_type: Type[GridObject]):
        self.data.append(object_type)

    def names(self) -> List[str]:
        return [object_type.__name__ for object_type in self.data]

    def from_name(self, name: str) -> Type[GridObject]:
        try:
            return next(
                object_type
                for object_type in self.data
                if object_type.__name__ == name
            )
        except StopIteration as error:
            raise ValueError(f'Unregistered GridObject `{name}`') from error


grid_object_registry = GridObjectRegistry()


class GridObjectMeta(abc.ABCMeta):
    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        # checks attribute existence at object instantiation
        obj.state_index
        obj.color
        obj.transparent
        obj.can_be_picked_up
        obj.blocks
        return obj


class GridObject(metaclass=GridObjectMeta):
    """A cell in the grid"""

    def __init__(self):
        self.state_index: int
        """State index of the object"""

        self.color: Color
        """Color of the object"""

        self.transparent: bool
        """Whether the agent can see _through_ the object"""

        self.can_be_picked_up: bool
        """Whether the agent can see pick up the object"""

        self.blocks: bool
        """Whether the object blocks the agent from moving on it"""

    def __init_subclass__(cls, *, register=True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register:
            grid_object_registry.register(cls)

    @classmethod
    def type_index(cls) -> int:
        return grid_object_registry.index(cls)

    @classmethod
    @abc.abstractmethod
    def can_be_represented_in_state(cls) -> bool:
        """Returns whether the state_index fully represents the object state"""

    @classmethod
    @abc.abstractmethod
    def num_states(cls) -> int:
        """Number of states this class can take on"""

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
    """object representing the absence of an object"""

    def __init__(self):
        super().__init__()
        self.state_index = 0
        self.color = Color.NONE
        self.transparent = False
        self.can_be_picked_up = False
        self.blocks = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Hidden(GridObject):
    """object representing an unobservable cell"""

    def __init__(self):
        super().__init__()
        self.state_index = 0
        self.color = Color.NONE
        self.transparent = False
        self.can_be_picked_up = False
        self.blocks = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return False

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Floor(GridObject):
    """Most basic object in the grid, represents empty cell"""

    def __init__(self):
        super().__init__()
        self.state_index = 0
        self.color = Color.NONE
        self.transparent = True
        self.can_be_picked_up = False
        self.blocks = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Wall(GridObject):
    """The (second) most basic object in the grid: blocking cell"""

    def __init__(self):
        super().__init__()
        self.state_index = 0
        self.color = Color.NONE
        self.transparent = False
        self.can_be_picked_up = False
        self.blocks = True

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

    def __init__(self, color: Color = Color.NONE):
        super().__init__()
        self.state_index = 0
        self.color = color
        self.transparent = True
        self.can_be_picked_up = False
        self.blocks = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

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

    class Status(enum.Enum):
        """open, closed or locked"""

        OPEN = 0
        CLOSED = enum.auto()
        LOCKED = enum.auto()

    def __init__(self, state: Door.Status, color: Color):
        super().__init__()
        self.state = state
        self.color = color
        self.can_be_picked_up = False

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
    def transparent(self) -> bool:
        return self.state is Door.Status.OPEN

    @property
    def blocks(self) -> bool:
        return self.state is not Door.Status.OPEN

    @property
    def is_open(self) -> bool:
        """returns whether the door is opened"""
        return self.state is Door.Status.OPEN

    @property
    def locked(self) -> bool:
        """returns whether the door is locked"""
        return self.state is Door.Status.LOCKED

    def __repr__(self):
        return f'{self.__class__.__name__}({self.state!s}, {self.color!s})'


class Key(GridObject):
    """A key opens a door with the same color"""

    def __init__(self, color: Color):
        super().__init__()
        self.state_index = 0
        self.color = color
        self.transparent = True
        self.can_be_picked_up = True
        self.blocks = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}({self.color!s})'


class MovingObstacle(GridObject):
    """An obstacle to be avoided that moves in the grid"""

    def __init__(self):
        super().__init__()
        self.state_index = 0
        self.color = Color.NONE
        self.transparent = True
        self.can_be_picked_up = False
        self.blocks = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class Box(GridObject):
    """A box which can be broken and may contain another object"""

    def __init__(self, content: GridObject):
        super().__init__()

        """Boxes have no special status or color"""
        if isinstance(content, (NoneGridObject, Hidden)):
            raise ValueError('Box cannot contain NoneGridObject or Hidden')

        self.content = content

        self.state_index = 0
        self.color = Color.NONE
        self.transparent = True
        self.can_be_picked_up = False
        self.blocks = True

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return False

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}({self.content!r})'


class Telepod(GridObject):
    """A teleportation pod"""

    def __init__(self, color: Color):
        super().__init__()
        self.state_index = 0
        self.color = color
        self.transparent = True
        self.can_be_picked_up = False
        self.blocks = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}({self.color!s})'


class Beacon(GridObject):
    """A beacon to attract attention"""

    def __init__(self, color: Color):
        self.state_index = 0
        self.color = color
        self.transparent = True
        self.can_be_picked_up = False
        self.blocks = False

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}({self.color!s})'


GridObjectFactory = Callable[[], GridObject]
"""Signature for a function that instantiates grid objects on call"""
