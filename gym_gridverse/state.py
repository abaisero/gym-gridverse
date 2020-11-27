from __future__ import annotations

from .info import Agent, Grid


class State:
    """A state is represented by two pieces: a grid and an agent

    The grid :py:class:`~gym_gridverse.info.Grid` is a two-dimensional array of
    :py:class:`~gym_gridverse.grid_object.GridObject`. The
    :py:class:`~gym_gridverse.info.Agent` describes the agent's location,
    orientation and holding item.

    This class offers little functionality, and basically is just a holder for
    those two components.
    """

    def __init__(self, grid: Grid, agent: Agent):
        self.grid = grid
        self.agent = agent

    def __eq__(self, other):
        if isinstance(other, State):
            return self.grid == other.grid and self.agent == other.agent
        return NotImplemented
