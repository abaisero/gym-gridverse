import abc
from typing import Dict

import numpy as np


class Representation(metaclass=abc.ABCMeta):
    """Base interface for state and observation representation"""

    @property
    @abc.abstractmethod
    def space(self) -> Dict[str, np.ndarray]:
        """range of values the representation can return

        returns a string -> numpy array of max values, e.g:
        same shape as `convert` but returns max values
        """

    @abc.abstractmethod
    def convert(self, x) -> Dict[str, np.ndarray]:
        """represents x"""
