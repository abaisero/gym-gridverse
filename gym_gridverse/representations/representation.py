import abc
from typing import Dict

import numpy as np
from gym_gridverse.grid_object import GridObject
from gym_gridverse.observation import Observation
from gym_gridverse.state import State


class Representation(metaclass=abc.ABCMeta):
    """Base interface for state, observation and object representation"""

    @property
    @abc.abstractmethod
    def space(self) -> Dict[str, np.ndarray]:
        """range of values the representation can return

        returns a string -> numpy array of max values, e.g:
        same shape as `convert` but returns max values
        """


class StateRepresentation(Representation):
    """Base interface for state representations: enforces `convert`"""

    @abc.abstractmethod
    def convert(self, s: State) -> Dict[str, np.ndarray]:
        """returns state `s` representation as str -> array dict"""


class ObservationRepresentation(Representation):
    """Base interface for observation representations: enforces `convert`"""

    @abc.abstractmethod
    def convert(self, o: Observation) -> Dict[str, np.ndarray]:
        """returns observation `o` representation as str -> array dict"""


class GridObjectRepresentation(Representation):
    """Base interface for observation representations: enforces `convert`"""

    @abc.abstractmethod
    def convert(self, obj: GridObject) -> Dict[str, np.ndarray]:
        """returns grid object `obj` representation as str -> array dict"""
