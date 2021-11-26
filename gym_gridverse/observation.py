"""Defines the Observation class"""
from dataclasses import dataclass

from gym_gridverse.agent import Agent
from gym_gridverse.grid import Grid


@dataclass(frozen=True)
class Observation:
    """An observation is a pure data-class containing ``grid`` and ``agent``
    components.

    This class offers little functionality, and is just a holder for grid and
    agent.  The grid is an instance of class
    :py:class:`~gym_gridverse.grid.Grid`, which represents the gridworld from
    the agent's POV.  The agent is an instance of class
    :py:class:`~gym_gridverse.agent.Agent`, which represents the agent's
    location, orientation, and held item, all from the agent's POV.
    """

    grid: Grid
    agent: Agent
