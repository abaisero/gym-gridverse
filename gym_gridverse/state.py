from __future__ import annotations

from .info import Agent, Grid


class State:
    def __init__(self, grid: Grid, agent: Agent):
        self.grid = grid
        self.agent = agent

    def __eq__(self, other):
        if isinstance(other, State):
            return self.grid == other.grid and self.agent == other.agent
        return NotImplemented
