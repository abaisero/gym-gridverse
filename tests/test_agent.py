"""Tests agent related functionality"""


import pytest

from gym_gridverse.agent import Agent
from gym_gridverse.geometry import Area, Orientation, Position, PositionOrTuple


@pytest.mark.parametrize(
    'position,orientation,expected',
    [
        ((0, 0), Orientation.N, Area((-6, 0), (-3, 3))),
        ((0, 0), Orientation.S, Area((0, 6), (-3, 3))),
        ((0, 0), Orientation.E, Area((-3, 3), (0, 6))),
        ((0, 0), Orientation.W, Area((-3, 3), (-6, 0))),
        ((1, 2), Orientation.N, Area((-5, 1), (-1, 5))),
        ((1, 2), Orientation.S, Area((1, 7), (-1, 5))),
        ((1, 2), Orientation.E, Area((-2, 4), (2, 8))),
        ((1, 2), Orientation.W, Area((-2, 4), (-4, 2))),
    ],
)
def test_get_pov_area(
    position: PositionOrTuple, orientation: Orientation, expected: Area
):
    relative_area = Area((-6, 0), (-3, 3))
    agent = Agent(position, orientation)
    assert agent.get_pov_area(relative_area) == expected


@pytest.mark.parametrize(
    'position,orientation,delta_position,expected',
    [
        ((0, 0), Orientation.N, Position(1, -1), (1, -1)),
        ((0, 0), Orientation.S, Position(1, -1), (-1, 1)),
        ((0, 0), Orientation.E, Position(1, -1), (-1, -1)),
        ((0, 0), Orientation.W, Position(1, -1), (1, 1)),
        ((1, 2), Orientation.N, Position(2, -2), (3, 0)),
        ((1, 2), Orientation.S, Position(2, -2), (-1, 4)),
        ((1, 2), Orientation.E, Position(2, -2), (-1, 0)),
        ((1, 2), Orientation.W, Position(2, -2), (3, 4)),
    ],
)
def test_agent_position_relative(
    position: PositionOrTuple,
    orientation: Orientation,
    delta_position: Position,
    expected: PositionOrTuple,
):
    agent = Agent(position, orientation)
    assert agent.position_relative(delta_position) == expected


@pytest.mark.parametrize(
    'position,orientation,expected',
    [
        ((0, 0), Orientation.N, (-1, 0)),
        ((0, 0), Orientation.S, (1, 0)),
        ((0, 0), Orientation.E, (0, 1)),
        ((0, 0), Orientation.W, (0, -1)),
        ((1, 2), Orientation.N, (0, 2)),
        ((1, 2), Orientation.S, (2, 2)),
        ((1, 2), Orientation.E, (1, 3)),
        ((1, 2), Orientation.W, (1, 1)),
    ],
)
def test_agent_position_in_front(
    position: PositionOrTuple,
    orientation: Orientation,
    expected: PositionOrTuple,
):
    agent = Agent(position, orientation)
    assert agent.position_in_front() == expected
