import os
import sys
from unittest.mock import MagicMock, PropertyMock

import pytest

from gym_gridverse.action import Action
from gym_gridverse.state import State

# avoid discovering tests in setup.py
collect_ignore = ["setup.py"]


# setup PYTHONPATH to allow import custom modules in yaml/
@pytest.fixture(scope='session', autouse=True)
def execute_before_tests():
    base = os.path.dirname(__file__)
    sys.path.append(f'{base}/yaml')


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
    def _forbidden_action_maker() -> Action:
        action = MagicMock()
        return action

    yield _forbidden_action_maker
