from __future__ import annotations

from typing import Optional

from .geometry import Area, Orientation, Position, PositionOrTuple
from .grid_object import GridObject, NoneGridObject


class Agent:
    """The agent part of the state in an environment

    A container for the:
        - :py:class:`~gym_gridverse.geometry.Position` of the agent
        - :py:class:`~gym_gridverse.geometry.Orientation` of the agent
        - :py:class:`~gym_gridverse.grid_object.GridObject` of the agent

    Adds some API functionality
    """

    def __init__(
        self,
        position: PositionOrTuple,
        orientation: Orientation,
        obj: Optional[GridObject] = None,
    ):
        """Creates the agent on `position` with `orientation` and holding `obj`

        Args:
            position (PositionOrTuple):
            orientation (Orientation):
            obj (Optional[GridObject]):
        """

        self.position = position  # type: ignore
        self.orientation = orientation
        self.obj: GridObject = NoneGridObject() if obj is None else obj

    @property
    def position(self) -> Position:
        return self._position

    @position.setter
    def position(self, position: PositionOrTuple):
        self._position = Position.from_position_or_tuple(position)

    def __eq__(self, other):
        if isinstance(other, Agent):
            return (
                self.position == other.position
                and self.orientation == other.orientation
                and self.obj == other.obj
            )

        return NotImplemented

    def position_relative(self, dpos: Position) -> Position:
        """get the absolute position from a delta position relative to the agent"""
        return self.position + dpos.rotate(self.orientation)

    def position_in_front(self) -> Position:
        """get the position in front of the agent"""
        return self.position_relative(Orientation.N.as_position())

    def get_pov_area(self, relative_area: Area) -> Area:
        """gets absolute area corresponding to given relative area

        The relative ares is relative to the agent's POV, with position (0, 0)
        representing the agent's position.  The absolute area is the relative
        ares translated and rotated such as to indicate the agent's POV in
        absolute state coordinates.
        """
        return relative_area.rotate(self.orientation).translate(self.position)

    def __hash__(self):
        return hash((self.position, self.orientation, self.obj))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.position!r}, {self.orientation!s})'
            if isinstance(self.obj, NoneGridObject)
            else f'{self.__class__.__name__}({self.position!r}, {self.orientation!s}, {self.obj!r})'
        )
