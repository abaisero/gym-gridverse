from typing import Tuple

import numpy as np

from .grid_object import GridObject


class Grid:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.__grid = np.empty(self.shape, dtype=object)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    def is_proper(self) -> bool:
        #  pylint: disable=not-an-iterable
        return all(x is not None for x in self.__grid.flat)

    # TODO what type is position?
    def __getitem__(self, position: Tuple[int, int]):
        return self.__grid[position]

    def __setitem__(self, position: Tuple[int, int], obj: GridObject):
        if not isinstance(obj, GridObject):
            TypeError('grid can only contain entities')

        self.__grid[position] = obj


class Agent:
    def __init__(self, position, orientation, obj: GridObject):
        self.position = position
        self.orientation = orientation
        self.obj = obj


class State:
    def __init__(self, grid: Grid, agent: Agent):
        self.grid = grid
        self.agent = agent


class Observation:
    def __init__(self, grid: Grid, agent: Agent):
        self.grid = grid
        self.agent = agent
