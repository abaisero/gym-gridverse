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
    def register(self, object_type: Type[GridObject]) -> Type[GridObject]:
        self.data.append(object_type)
        return object_type

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


# class GridObjectMeta(abc.ABCMeta):
#     def __call__(self, *args, **kwargs):
#         obj = super().__call__(*args, **kwargs)
#         # checks attribute existence at object instantiation
#         obj.state_index
#         obj.color
#         obj.blocks_movement
#         obj.blocks_vision
#         obj.holdable
#         return obj


# class GridObject(metaclass=GridObjectMeta):
class GridObject(metaclass=abc.ABCMeta):
    """A cell in the grid"""

    @property
    @abc.abstractmethod
    def state_index(self) -> int:
        """State index of the object"""

    @property
    @abc.abstractmethod
    def color(self) -> Color:
        """Color of the object"""

    @property
    @abc.abstractmethod
    def blocks_movement(self) -> bool:
        """Whether the object blocks the agent from moving on it"""

    @property
    @abc.abstractmethod
    def blocks_vision(self) -> bool:
        """Whether the object blocks the agent's vision."""

    @property
    @abc.abstractmethod
    def holdable(self) -> bool:
        """Whether the agent can see pick up the object"""

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
    """object representing an unobservable cell"""

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
    """Most basic object in the grid, represents empty cell"""

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
    """The (second) most basic object in the grid: blocking cell"""

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
    """A door in a grid.

    Can be `OPEN`, `CLOSED` or `LOCKED`.
    """

    color = Color.NONE
    holdable = False

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
    """A key opens a door with the same color"""

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
    """An obstacle to be avoided that moves in the grid"""

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
    """A box which can be broken and may contain another object"""

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
    """A teleportation pod"""

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
    """A beacon to attract attention"""

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


GridObjectFactory = Callable[[], GridObject]
"""Signature for a function that instantiates grid objects on call"""
