from typing import Iterator, Optional

import numpy as np

from .geometry import Orientation, Position, Shape
from .grid_object import Floor, GridObject


class Grid:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self._grid = np.array(
            [[Floor() for _ in range(width)] for _ in range(height)],
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented

        if self.shape != other.shape:
            return False

        for pos in self.positions():
            if self[pos] != other[pos]:
                return False

        return True

    @property
    def shape(self) -> Shape:
        return Shape(self.height, self.width)

    def __contains__(self, position: Position) -> bool:
        return 0 <= position.y < self.height and 0 <= position.x < self.width

    def _check_contains(self, position: Position):
        if position not in self:
            raise ValueError(f'Position {position} ')

    def positions(self) -> Iterator[Position]:
        for indices in iter(np.indices(self.shape).reshape(2, -1).T):
            yield Position(*indices)

    def get_position(self, x: GridObject) -> Position:
        for position in self.positions():
            if self[position] is x:
                return position

        raise ValueError(f'GridObject {x} not found')

    def __getitem__(self, position: Position) -> GridObject:
        return self._grid[position]

    def __setitem__(self, position: Position, obj: GridObject):
        if not isinstance(obj, GridObject):
            TypeError('grid can only contain entities')
        self._grid[position] = obj

    def swap(self, p: Position, q: Position):
        self._check_contains(p)
        self._check_contains(q)
        self[p], self[q] = self[q], self[p]


class Agent:
    def __init__(
        self,
        position: Position,
        orientation: Orientation,
        obj: Optional[GridObject] = None,
    ):
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
