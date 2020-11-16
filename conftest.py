from unittest.mock import MagicMock, PropertyMock

import pytest

from gym_gridverse.actions import Actions
from gym_gridverse.state import State

# avoid discovering tests in setup.py
collect_ignore = ["setup.py"]


@pytest.fixture
def forbidden_state_maker():
    grids = []
    agents = []

    def _forbidden_state_maker() -> State:
        state = MagicMock()
        type(state).grid = grid = PropertyMock()
        type(state).agent = agent = PropertyMock()

        grids.append(grid)
        agents.append(agent)
        return state

    yield _forbidden_state_maker

    for grid in grids:
        grid.assert_not_called()

    for agent in agents:
        agent.assert_not_called()


@pytest.fixture
def forbidden_action_maker():
    def _forbidden_action_maker() -> Actions:
        action = MagicMock()
        return action

    yield _forbidden_action_maker
