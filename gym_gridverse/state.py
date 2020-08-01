from __future__ import annotations

from .info import Agent, Grid


class State:
    def __init__(self, grid: Grid, agent: Agent):
        self.grid = grid
        self.agent = agent
