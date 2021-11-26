from __future__ import annotations

from typing import Optional

from .geometry import Orientation, Position, Transform
from .grid_object import GridObject, NoneGridObject


class Agent:
    """Information relative to the agent.

    NOTE:  This does not necessarily represent the true full state of the
    agent;  e.g., the agent field of an observation objects, would only contain
    the observable versions of the agent's state.
    """

    def __init__(
        self,
        position: Position,
        orientation: Orientation,
        grid_object: Optional[GridObject] = None,
    ):
        """Creates the agent at `position` with `orientation` and holding `grid_object`.

        Args:
            position (Position): position of the agent relative to some area.
            orientation (Orientation): orientation of the agent relative to some area.
            grid_object (Optional[GridObject]): object held by the agent.
        """

        self.transform = Transform(position, orientation)
        self.grid_object: GridObject = (
            NoneGridObject() if grid_object is None else grid_object
        )

    def front(self) -> Position:
        return self.transform * Position.from_orientation(Orientation.F)

    @property
    def position(self) -> Position:
        return self.transform.position

    @position.setter
    def position(self, position: Position):
        self.transform.position = position

    @property
    def orientation(self) -> Orientation:
        return self.transform.orientation

    @orientation.setter
    def orientation(self, orientation: Orientation):
        self.transform.orientation = orientation

    def __eq__(self, other):
        if isinstance(other, Agent):
            return (
                self.transform == other.transform
                and self.grid_object == other.grid_object
            )

        return NotImplemented

    def __hash__(self):
        return hash((self.transform, self.grid_object))

    def __repr__(self):
        # TODO: test
        return (
            f'{self.__class__.__name__}({self.position!r}, {self.orientation!s})'
            if isinstance(self.grid_object, NoneGridObject)
            else f'{self.__class__.__name__}({self.position!r}, {self.orientation!s}, {self.grid_object!r})'
        )
