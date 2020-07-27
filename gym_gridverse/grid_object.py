""" Every cell in the grid is represented by a grid objects """

import abc
import enum

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

    @property
    def object_type(self) -> int:
        """ Returns the type of the object """
        return self._type  # NOTE: magically assumed set through registration

    @property
    @abc.abstractmethod
    def state(self) -> int:
        """ Returns the state of the object """

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
    def update(self, state) -> None:
        """ Optional behavior of the object """

    @abc.abstractmethod
    def actuate(self, state) -> None:
        """ The (optional) behavior upon actuation """

    def as_array(self):
        """ A (3,) array representation of the object """
        return np.array([self.object_type, self.state, int(self.color.value)])


class Floor(GridObject):
    """ Most basic object in the grid, represents empty cell """

    _type = 0  # TODO: set automatically through registration

    def __init__(self):
        pass

    @property
    def state(self) -> int:
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

    def update(self, state) -> None:
        pass

    def actuate(self, state) -> None:
        pass
