from __future__ import annotations

from typing import Optional

from .geometry import Orientation, Pose, Position
from .grid_object import GridObject, NoneGridObject


class Agent:
    """The agent part of the state in an environment

    A container for the:
        - :py:class:`~gym_gridverse.geometry.Position` of the agent
        - :py:class:`~gym_gridverse.geometry.Orientation` of the agent
        - :py:class:`~gym_gridverse.grid_object.GridObject` of the agent
    """

    def __init__(
        self,
        position: Position,
        orientation: Orientation,
        obj: Optional[GridObject] = None,
    ):
        """Creates the agent on `position` with `orientation` and holding `obj`

        Args:
            position (Position):
            orientation (Orientation):
            obj (Optional[GridObject]):
        """

        self.pose = Pose(position, orientation)
        self.obj: GridObject = NoneGridObject() if obj is None else obj

    @property
    def position(self) -> Position:
        return self.pose.position

    @position.setter
    def position(self, position: Position):
        self.pose.position = position

    @property
    def orientation(self) -> Orientation:
        return self.pose.orientation

    @orientation.setter
    def orientation(self, orientation: Orientation):
        self.pose.orientation = orientation

    def __eq__(self, other):
        if isinstance(other, Agent):
            return self.pose == other.pose and self.obj == other.obj

        return NotImplemented

    def __hash__(self):
        return hash((self.pose, self.obj))

    def __repr__(self):
        # TODO: test
        return (
            f'{self.__class__.__name__}({self.position!r}, {self.orientation!s})'
            if isinstance(self.obj, NoneGridObject)
            else f'{self.__class__.__name__}({self.position!r}, {self.orientation!s}, {self.obj!r})'
        )
