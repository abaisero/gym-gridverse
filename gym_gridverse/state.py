"""Defines the State class"""
from dataclasses import dataclass

from gym_gridverse.agent import Agent
from gym_gridverse.grid import Grid


@dataclass(frozen=True)
class State:
    """A state is represented by two pieces: a grid and an agent

    The grid :py:class:`~gym_gridverse.grid.Grid` is a two-dimensional array of
    :py:class:`~gym_gridverse.grid_object.GridObject`. The
    :py:class:`~gym_gridverse.agent.Agent` describes the agent's location,
    orientation and holding item.

    This class offers little functionality, and basically is just a holder for
    those two components.
    """

    grid: Grid
    agent: Agent
