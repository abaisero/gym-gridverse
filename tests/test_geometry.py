import math
from typing import Sequence

import pytest

from gym_gridverse.geometry import (
    Area,
    DeltaPosition,
    Orientation,
    Position,
    PositionOrTuple,
    get_manhattan_boundary,
)


@pytest.mark.parametrize(
    'area,expected', [(Area((0, 1), (0, 2)), 2), (Area((-1, 1), (-2, 2)), 3)],
)
def test_area_height(area: Area, expected: int):
    assert area.height == expected


@pytest.mark.parametrize(
    'area,expected', [(Area((0, 1), (0, 2)), 3), (Area((-1, 1), (-2, 2)), 5)],
)
def test_area_width(area: Area, expected: int):
    assert area.width == expected


@pytest.mark.parametrize(
    'area,expected',
    [(Area((0, 1), (0, 2)), (0, 0)), (Area((-1, 1), (-2, 2)), (-1, -2))],
)
def test_area_top_left(area: Area, expected: PositionOrTuple):
    assert area.top_left == expected


@pytest.mark.parametrize(
    'area,expected',
    [(Area((0, 1), (0, 2)), (0, 2)), (Area((-1, 1), (-2, 2)), (-1, 2))],
)
def test_area_top_right(area: Area, expected: PositionOrTuple):
    assert area.top_right == expected


@pytest.mark.parametrize(
    'area,expected',
    [(Area((0, 1), (0, 2)), (1, 0)), (Area((-1, 1), (-2, 2)), (1, -2))],
)
def test_area_bottom_left(area: Area, expected: PositionOrTuple):
    assert area.bottom_left == expected


@pytest.mark.parametrize(
    'area,expected',
    [(Area((0, 1), (0, 2)), (1, 2)), (Area((-1, 1), (-2, 2)), (1, 2))],
)
def test_area_bottom_right(area: Area, expected: PositionOrTuple):
    assert area.bottom_right == expected


@pytest.mark.parametrize(
    'area,position,expected',
    [
        (Area((0, 1), (0, 2)), (0, 0), True),
        (Area((0, 1), (0, 2)), (-1, 0), False),
        (Area((0, 1), (0, 2)), (0, -1), False),
        #
        (Area((0, 1), (0, 2)), (1, 2), True),
        (Area((0, 1), (0, 2)), (2, 2), False),
        (Area((0, 1), (0, 2)), (1, 3), False),
        #
        (Area((-1, 1), (-2, 2)), (-1, -2), True),
        (Area((-1, 1), (-2, 2)), (-2, -2), False),
        (Area((-1, 1), (-2, 2)), (-1, -3), False),
        #
        (Area((-1, 1), (-2, 2)), (1, 2), True),
        (Area((-1, 1), (-2, 2)), (2, 2), False),
        (Area((-1, 1), (-2, 2)), (1, 3), False),
    ],
)
def test_area_contains(area: Area, position: PositionOrTuple, expected: bool):
    assert area.contains(position) == expected


@pytest.mark.parametrize(
    'area,position,expected',
    [
        (Area((0, 1), (0, 2)), (1, -1), Area((1, 2), (-1, 1))),
        (Area((0, 1), (0, 2)), (-1, 1), Area((-1, 0), (1, 3))),
        #
        (Area((-1, 1), (-2, 2)), (1, -1), Area((0, 2), (-3, 1))),
        (Area((-1, 1), (-2, 2)), (-1, 1), Area((-2, 0), (-1, 3))),
    ],
)
def test_area_translate(area: Area, position: PositionOrTuple, expected: Area):
    assert area.translate(position) == expected


@pytest.mark.parametrize(
    'area,orientation,expected',
    [
        (Area((0, 1), (0, 2)), Orientation.N, Area((0, 1), (0, 2))),
        (Area((0, 1), (0, 2)), Orientation.S, Area((-1, 0), (-2, 0))),
        (Area((0, 1), (0, 2)), Orientation.E, Area((0, 2), (-1, 0))),
        (Area((0, 1), (0, 2)), Orientation.W, Area((-2, 0), (0, 1))),
        #
        (Area((-1, 1), (-2, 2)), Orientation.N, Area((-1, 1), (-2, 2))),
        (Area((-1, 1), (-2, 2)), Orientation.S, Area((1, -1), (2, -2))),
        (Area((-1, 1), (-2, 2)), Orientation.E, Area((-2, 2), (-1, 1))),
        (Area((-1, 1), (-2, 2)), Orientation.W, Area((-2, 2), (-1, 1))),
    ],
)
def test_area_rotate(area: Area, orientation: Orientation, expected: Area):
    assert area.rotate(orientation) == expected


@pytest.mark.parametrize(
    'area1,area2,expected',
    [
        (Area((0, 1), (0, 2)), Area((0, 1), (0, 2)), True),
        (Area((0, 1), (0, 2)), Area((-1, 1), (-2, 2)), False),
        #
        (Area((-1, 1), (-2, 2)), Area((0, 1), (0, 2)), False),
        (Area((-1, 1), (-2, 2)), Area((-1, 1), (-2, 2)), True),
    ],
)
def test_area_eq(area1: Area, area2: Area, expected: bool):
    assert (area1 == area2) == expected


@pytest.mark.parametrize(
    'orientation,dist,delta_position',
    [
        (Orientation.N, 1, DeltaPosition(-1, 0)),
        (Orientation.N, 2, DeltaPosition(-2, 0)),
        #
        (Orientation.S, 1, DeltaPosition(1, 0)),
        (Orientation.S, 2, DeltaPosition(2, 0)),
        #
        (Orientation.E, 1, DeltaPosition(0, 1)),
        (Orientation.E, 2, DeltaPosition(0, 2)),
        #
        (Orientation.W, 1, DeltaPosition(0, -1)),
        (Orientation.W, 2, DeltaPosition(0, -2)),
    ],
)
def test_orientation_as_delta_position(
    orientation: Orientation, dist: int, delta_position: DeltaPosition
):
    assert orientation.as_delta_position(dist) == delta_position


@pytest.mark.parametrize(
    'position,distance,expected',
    [
        ((2, 2), 1, [(1, 2), (2, 3), (3, 2), (2, 1)]),
        (
            (4, 3),
            2,
            [(2, 3), (3, 4), (4, 5), (5, 4), (6, 3), (5, 2), (4, 1), (3, 2)],
        ),
    ],
)
def test_manhattan_boundary(
    position: PositionOrTuple,
    distance: int,
    expected: Sequence[PositionOrTuple],
):
    boundary = get_manhattan_boundary(position, distance)
    assert len(boundary) == len(expected)
    assert all(expected_position in boundary for expected_position in expected)


@pytest.mark.parametrize('y', [-10, 0, 10])
@pytest.mark.parametrize('x', [-10, 0, 10])
def test_position_from_position_or_tuple(y: int, x: int):
    position = Position(y, x)
    position_from_position = Position.from_position_or_tuple(position)
    assert isinstance(position_from_position, Position)
    assert position_from_position is position

    position_from_tuple = Position.from_position_or_tuple((y, x))
    assert isinstance(position_from_tuple, Position)

    assert position_from_position == position_from_tuple


@pytest.mark.parametrize(
    'pos1,pos2,expected',
    [
        (Position(y1, x1), Position(y2, x2), Position(y1 + y2, x1 + x2))
        for x1 in range(2)
        for y1 in range(2)
        for x2 in range(2)
        for y2 in range(2)
    ],
)
def test_position_add(pos1: Position, pos2: Position, expected: Position):
    assert Position.add(pos1, pos2) == expected


@pytest.mark.parametrize(
    'pos1,pos2,expected',
    [
        (Position(y1, x1), Position(y2, x2), Position(y1 - y2, x1 - x2))
        for x1 in range(2)
        for y1 in range(2)
        for x2 in range(2)
        for y2 in range(2)
    ],
)
def test_position_subtract(pos1: Position, pos2: Position, expected: Position):
    assert Position.subtract(pos1, pos2) == expected


@pytest.mark.parametrize(
    'pos1,pos2,expected',
    [
        (Position(0, 0), Position(0, 0), 0.0),
        (Position(0, 0), Position(0, 1), 1.0),
        (Position(0, 0), Position(1, 1), 2.0),
        (Position(0, 1), Position(1, 1), 1.0),
        (Position(1, 1), Position(1, 1), 0.0),
        # diagonal
        (Position(0, 0), Position(0, 0), 0.0),
        (Position(0, 0), Position(1, 1), 2.0),
        (Position(0, 0), Position(2, 2), 4.0),
        (Position(0, 0), Position(3, 3), 6.0),
    ],
)
def test_position_manhattan_distance(
    pos1: Position, pos2: Position, expected: float
):
    assert Position.manhattan_distance(pos1, pos2) == expected


@pytest.mark.parametrize(
    'pos1,pos2,expected',
    [
        (Position(0, 0), Position(0, 0), 0.0),
        (Position(0, 0), Position(0, 1), 1.0),
        (Position(0, 0), Position(1, 1), math.sqrt(2.0)),
        (Position(0, 1), Position(1, 1), 1.0),
        (Position(1, 1), Position(1, 1), 0.0),
        # diagonal
        (Position(0, 0), Position(0, 0), 0.0),
        (Position(0, 0), Position(1, 1), math.sqrt(2.0)),
        (Position(0, 0), Position(2, 2), math.sqrt(8.0)),
        (Position(0, 0), Position(3, 3), math.sqrt(18.0)),
    ],
)
def test_position_euclidean_distance(
    pos1: Position, pos2: Position, expected: float
):
    assert Position.euclidean_distance(pos1, pos2) == expected


@pytest.mark.parametrize(
    'delta_position,orientation,expected',
    [
        # y basis
        (DeltaPosition(1, 0), Orientation.N, DeltaPosition(1, 0)),
        (DeltaPosition(1, 0), Orientation.S, DeltaPosition(-1, 0)),
        (DeltaPosition(1, 0), Orientation.E, DeltaPosition(0, -1)),
        (DeltaPosition(1, 0), Orientation.W, DeltaPosition(0, 1)),
        # x basis
        (DeltaPosition(0, 1), Orientation.N, DeltaPosition(0, 1)),
        (DeltaPosition(0, 1), Orientation.S, DeltaPosition(0, -1)),
        (DeltaPosition(0, 1), Orientation.E, DeltaPosition(1, 0)),
        (DeltaPosition(0, 1), Orientation.W, DeltaPosition(-1, 0)),
        # others
        (DeltaPosition(1, 2), Orientation.N, DeltaPosition(1, 2)),
        (DeltaPosition(1, 2), Orientation.S, DeltaPosition(-1, -2)),
        (DeltaPosition(1, 2), Orientation.E, DeltaPosition(2, -1)),
        (DeltaPosition(1, 2), Orientation.W, DeltaPosition(-2, 1)),
    ],
)
def test_delta_position_rotate_basis(
    delta_position: DeltaPosition,
    orientation: Orientation,
    expected: DeltaPosition,
):
    assert delta_position.rotate(orientation) == expected
