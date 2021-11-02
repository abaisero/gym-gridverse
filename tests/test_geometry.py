import math
from typing import Sequence

import pytest

from gym_gridverse.geometry import (
    Area,
    Orientation,
    Position,
    StrideDirection,
    Transform,
    diagonal_strides,
    get_manhattan_boundary,
)


def T(y: int, x: int, orientation: Orientation) -> Transform:
    return Transform(Position(y, x), orientation)


P = Position
O_ = Orientation
A = Area


@pytest.mark.parametrize(
    'area,expected',
    [
        (A((0, 1), (0, 2)), 2),
        (A((-1, 1), (-2, 2)), 3),
    ],
)
def test_area_height(area: Area, expected: int):
    assert area.height == expected


@pytest.mark.parametrize(
    'area,expected',
    [
        (A((0, 1), (0, 2)), 3),
        (A((-1, 1), (-2, 2)), 5),
    ],
)
def test_area_width(area: Area, expected: int):
    assert area.width == expected


@pytest.mark.parametrize(
    'area,expected',
    [
        (A((0, 1), (0, 2)), P(0, 0)),
        (A((-1, 1), (-2, 2)), P(-1, -2)),
    ],
)
def test_area_top_left(area: Area, expected: Position):
    assert area.top_left == expected


@pytest.mark.parametrize(
    'area,expected',
    [
        (A((0, 1), (0, 2)), P(0, 2)),
        (A((-1, 1), (-2, 2)), P(-1, 2)),
    ],
)
def test_area_top_right(area: Area, expected: Position):
    assert area.top_right == expected


@pytest.mark.parametrize(
    'area,expected',
    [
        (A((0, 1), (0, 2)), P(1, 0)),
        (A((-1, 1), (-2, 2)), P(1, -2)),
    ],
)
def test_area_bottom_left(area: Area, expected: Position):
    assert area.bottom_left == expected


@pytest.mark.parametrize(
    'area,expected',
    [
        (A((0, 1), (0, 2)), P(1, 2)),
        (A((-1, 1), (-2, 2)), P(1, 2)),
    ],
)
def test_area_bottom_right(area: Area, expected: Position):
    assert area.bottom_right == expected


@pytest.mark.parametrize(
    'area,position,expected',
    [
        (A((0, 1), (0, 2)), P(0, 0), True),
        (A((0, 1), (0, 2)), P(-1, 0), False),
        (A((0, 1), (0, 2)), P(0, -1), False),
        #
        (A((0, 1), (0, 2)), P(1, 2), True),
        (A((0, 1), (0, 2)), P(2, 2), False),
        (A((0, 1), (0, 2)), P(1, 3), False),
        #
        (A((-1, 1), (-2, 2)), P(-1, -2), True),
        (A((-1, 1), (-2, 2)), P(-2, -2), False),
        (A((-1, 1), (-2, 2)), P(-1, -3), False),
        #
        (A((-1, 1), (-2, 2)), P(1, 2), True),
        (A((-1, 1), (-2, 2)), P(2, 2), False),
        (A((-1, 1), (-2, 2)), P(1, 3), False),
    ],
)
def test_area_contains(area: Area, position: Position, expected: bool):
    assert area.contains(position) == expected


@pytest.mark.parametrize(
    'area,position,expected',
    [
        (A((0, 1), (0, 2)), P(1, -1), A((1, 2), (-1, 1))),
        (A((0, 1), (0, 2)), P(-1, 1), A((-1, 0), (1, 3))),
        #
        (A((-1, 1), (-2, 2)), P(1, -1), A((0, 2), (-3, 1))),
        (A((-1, 1), (-2, 2)), P(-1, 1), A((-2, 0), (-1, 3))),
    ],
)
def test_area_add_position(area: Area, position: Position, expected: Area):
    assert area + position == position + area == expected


@pytest.mark.parametrize(
    'area,orientation,expected',
    [
        (A((0, 1), (0, 2)), O_.N, A((0, 1), (0, 2))),
        (A((0, 1), (0, 2)), O_.S, A((-1, 0), (-2, 0))),
        (A((0, 1), (0, 2)), O_.E, A((0, 2), (-1, 0))),
        (A((0, 1), (0, 2)), O_.W, A((-2, 0), (0, 1))),
        #
        (A((-1, 1), (-2, 2)), O_.N, A((-1, 1), (-2, 2))),
        (A((-1, 1), (-2, 2)), O_.S, A((-1, 1), (-2, 2))),
        (A((-1, 1), (-2, 2)), O_.E, A((-2, 2), (-1, 1))),
        (A((-1, 1), (-2, 2)), O_.W, A((-2, 2), (-1, 1))),
    ],
)
def test_area_mul_orientation(
    area: Area, orientation: Orientation, expected: Area
):
    assert area * orientation == orientation * area == expected


@pytest.mark.parametrize(
    'area1,area2,expected',
    [
        (A((0, 1), (0, 2)), A((0, 1), (0, 2)), True),
        (A((0, 1), (0, 2)), A((-1, 1), (-2, 2)), False),
        #
        (A((-1, 1), (-2, 2)), A((0, 1), (0, 2)), False),
        (A((-1, 1), (-2, 2)), A((-1, 1), (-2, 2)), True),
    ],
)
def test_area_eq(area1: Area, area2: Area, expected: bool):
    assert (area1 == area2) == expected


@pytest.mark.parametrize(
    'orientation,dist,delta_position',
    [
        (O_.N, 1, P(-1, 0)),
        (O_.N, 2, P(-2, 0)),
        #
        (O_.S, 1, P(1, 0)),
        (O_.S, 2, P(2, 0)),
        #
        (O_.E, 1, P(0, 1)),
        (O_.E, 2, P(0, 2)),
        #
        (O_.W, 1, P(0, -1)),
        (O_.W, 2, P(0, -2)),
    ],
)
def test_orientation_as_position(
    orientation: Orientation, dist: int, delta_position: Position
):
    assert orientation.as_position(dist) == delta_position


@pytest.mark.parametrize(
    'position,distance,expected',
    [
        (
            P(2, 2),
            1,
            [P(1, 2), P(2, 3), P(3, 2), P(2, 1)],
        ),
        (
            P(4, 3),
            2,
            [
                P(2, 3),
                P(3, 4),
                P(4, 5),
                P(5, 4),
                P(6, 3),
                P(5, 2),
                P(4, 1),
                P(3, 2),
            ],
        ),
    ],
)
def test_manhattan_boundary(
    position: Position,
    distance: int,
    expected: Sequence[Position],
):
    boundary = get_manhattan_boundary(position, distance)
    assert len(boundary) == len(expected)
    assert all(expected_position in boundary for expected_position in expected)


@pytest.mark.parametrize(
    'pos1,pos2,expected',
    [
        (P(y1, x1), P(y2, x2), P(y1 + y2, x1 + x2))
        for x1 in range(2)
        for y1 in range(2)
        for x2 in range(2)
        for y2 in range(2)
    ],
)
def test_position_add(pos1: Position, pos2: Position, expected: Position):
    assert pos1 + pos2 == pos2 + pos1 == expected


@pytest.mark.parametrize(
    'pos1,pos2,expected',
    [
        (P(y1, x1), P(y2, x2), P(y1 - y2, x1 - x2))
        for x1 in range(2)
        for y1 in range(2)
        for x2 in range(2)
        for y2 in range(2)
    ],
)
def test_position_sub(pos1: Position, pos2: Position, expected: Position):
    assert pos1 - pos2 == expected


@pytest.mark.parametrize(
    'pos,expected',
    [(P(y, x), P(-y, -x)) for x in range(2) for y in range(2)],
)
def test_position_neg(pos: Position, expected: Position):
    assert -pos == expected


@pytest.mark.parametrize(
    'pos1,pos2,expected',
    [
        (P(0, 0), P(0, 0), 0.0),
        (P(0, 0), P(0, 1), 1.0),
        (P(0, 0), P(1, 1), 2.0),
        (P(0, 1), P(1, 1), 1.0),
        (P(1, 1), P(1, 1), 0.0),
        # diagonal
        (P(0, 0), P(0, 0), 0.0),
        (P(0, 0), P(1, 1), 2.0),
        (P(0, 0), P(2, 2), 4.0),
        (P(0, 0), P(3, 3), 6.0),
    ],
)
def test_position_manhattan_distance(
    pos1: Position, pos2: Position, expected: float
):
    assert Position.manhattan_distance(pos1, pos2) == expected


@pytest.mark.parametrize(
    'pos1,pos2,expected',
    [
        (P(0, 0), P(0, 0), 0.0),
        (P(0, 0), P(0, 1), 1.0),
        (P(0, 0), P(1, 1), math.sqrt(2.0)),
        (P(0, 1), P(1, 1), 1.0),
        (P(1, 1), P(1, 1), 0.0),
        # diagonal
        (P(0, 0), P(0, 0), 0.0),
        (P(0, 0), P(1, 1), math.sqrt(2.0)),
        (P(0, 0), P(2, 2), math.sqrt(8.0)),
        (P(0, 0), P(3, 3), math.sqrt(18.0)),
    ],
)
def test_position_euclidean_distance(
    pos1: Position, pos2: Position, expected: float
):
    assert Position.euclidean_distance(pos1, pos2) == expected


@pytest.mark.parametrize(
    'orientation,position,expected',
    [
        # y basis
        (O_.N, P(1, 0), P(1, 0)),
        (O_.S, P(1, 0), P(-1, 0)),
        (O_.E, P(1, 0), P(0, -1)),
        (O_.W, P(1, 0), P(0, 1)),
        #       x basis
        (O_.N, P(0, 1), P(0, 1)),
        (O_.S, P(0, 1), P(0, -1)),
        (O_.E, P(0, 1), P(1, 0)),
        (O_.W, P(0, 1), P(-1, 0)),
        #       others
        (O_.N, P(1, 2), P(1, 2)),
        (O_.S, P(1, 2), P(-1, -2)),
        (O_.E, P(1, 2), P(2, -1)),
        (O_.W, P(1, 2), P(-2, 1)),
    ],
)
def test_orientation_mul_position(
    orientation: Orientation,
    position: Position,
    expected: Position,
):
    assert orientation * position == position * orientation == expected


@pytest.mark.parametrize(
    'transform,position,expected',
    [
        (T(1, 1, O_.E), P(1, 1), P(2, 0)),
        (T(1, 1, O_.E), P(1, -1), P(0, 0)),
        (T(1, 1, O_.E), P(-1, 1), P(2, 2)),
        (T(1, 1, O_.E), P(-1, -1), P(0, 2)),
    ],
)
def test_transform_mul_position(
    transform: Transform, position: Position, expected: Position
):
    assert transform * position == position * transform == expected


@pytest.mark.parametrize(
    'transform,orientation,expected',
    [
        (T(1, 1, O_.E), O_.N, O_.E),
        (T(1, 1, O_.E), O_.E, O_.S),
        (T(1, 1, O_.E), O_.S, O_.W),
        (T(1, 1, O_.E), O_.W, O_.N),
    ],
)
def test_transform_mul_orientation(
    transform: Transform, orientation: Orientation, expected: Orientation
):
    assert transform * orientation == orientation * transform == expected


@pytest.mark.parametrize(
    't,s,expected',
    [
        (T(0, 0, O_.N), T(0, 0, O_.N), T(0, 0, O_.N)),
        (T(0, 0, O_.N), T(-1, 1, O_.E), T(-1, 1, O_.E)),
        (T(0, 0, O_.N), T(-2, 2, O_.S), T(-2, 2, O_.S)),
        (T(0, 0, O_.N), T(-3, 3, O_.W), T(-3, 3, O_.W)),
        #
        (T(-1, 1, O_.E), T(0, 0, O_.N), T(-1, 1, O_.E)),
        (T(-1, 1, O_.E), T(-1, 1, O_.E), T(0, 2, O_.S)),
        (T(-1, 1, O_.E), T(-2, 2, O_.S), T(1, 3, O_.W)),
        (T(-1, 1, O_.E), T(-3, 3, O_.W), T(2, 4, O_.N)),
        #
        (T(-2, 2, O_.S), T(0, 0, O_.N), T(-2, 2, O_.S)),
        (T(-2, 2, O_.S), T(-1, 1, O_.E), T(-1, 1, O_.W)),
        (T(-2, 2, O_.S), T(-2, 2, O_.S), T(0, 0, O_.N)),
        (T(-2, 2, O_.S), T(-3, 3, O_.W), T(1, -1, O_.E)),
        #
        (T(-3, 3, O_.W), T(0, 0, O_.N), T(-3, 3, O_.W)),
        (T(-3, 3, O_.W), T(-1, 1, O_.E), T(-4, 2, O_.N)),
        (T(-3, 3, O_.W), T(-2, 2, O_.S), T(-5, 1, O_.E)),
        (T(-3, 3, O_.W), T(-3, 3, O_.W), T(-6, 0, O_.S)),
    ],
)
def test_transform_mul_transform(
    t: Transform, s: Transform, expected: Transform
):
    assert t * s == expected


@pytest.mark.parametrize(
    'transform,expected',
    [
        (T(0, 0, O_.N), T(0, 0, O_.N)),
        (T(-1, 1, O_.E), T(1, 1, O_.W)),
        (T(-2, 2, O_.S), T(-2, 2, O_.S)),
        (T(-3, 3, O_.W), T(-3, -3, O_.E)),
    ],
)
def test_transform_neg(transform: Transform, expected: Transform):
    assert (-transform) == expected


@pytest.mark.parametrize(
    'transform',
    [
        T(0, 0, O_.N),
        T(-1, 1, O_.E),
        T(-2, 2, O_.S),
        T(-3, 3, O_.W),
    ],
)
def test_transform_neg_identity(transform: Transform):
    assert transform * (-transform) == -transform * transform == T(0, 0, O_.N)


@pytest.mark.parametrize(
    'transform,area,expected',
    [
        (T(0, 0, O_.N), Area((1, 2), (1, 2)), Area((1, 2), (1, 2))),
        (T(0, 0, O_.N), Area((1, 2), (2, 4)), Area((1, 2), (2, 4))),
        (T(0, 0, O_.N), Area((2, 4), (1, 2)), Area((2, 4), (1, 2))),
        (T(0, 0, O_.N), Area((2, 4), (2, 4)), Area((2, 4), (2, 4))),
        #
        (T(-1, 1, O_.E), Area((1, 2), (1, 2)), Area((0, 1), (-1, 0))),
        (T(-1, 1, O_.E), Area((1, 2), (2, 4)), Area((1, 3), (-1, 0))),
        (T(-1, 1, O_.E), Area((2, 4), (1, 2)), Area((0, 1), (-3, -1))),
        (T(-1, 1, O_.E), Area((2, 4), (2, 4)), Area((1, 3), (-3, -1))),
        #
        (T(-2, 2, O_.S), Area((1, 2), (1, 2)), Area((-4, -3), (0, 1))),
        (T(-2, 2, O_.S), Area((1, 2), (2, 4)), Area((-4, -3), (-2, 0))),
        (T(-2, 2, O_.S), Area((2, 4), (1, 2)), Area((-6, -4), (0, 1))),
        (T(-2, 2, O_.S), Area((2, 4), (2, 4)), Area((-6, -4), (-2, 0))),
        #
        (T(-3, 3, O_.W), Area((1, 2), (1, 2)), Area((-5, -4), (4, 5))),
        (T(-3, 3, O_.W), Area((1, 2), (2, 4)), Area((-7, -5), (4, 5))),
        (T(-3, 3, O_.W), Area((2, 4), (1, 2)), Area((-5, -4), (5, 7))),
        (T(-3, 3, O_.W), Area((2, 4), (2, 4)), Area((-7, -5), (5, 7))),
    ],
)
def test_transform_mul_area(transform: Transform, area: Area, expected: Area):
    assert transform * area == expected


@pytest.mark.parametrize(
    'area,stride_direction,expected',
    [
        (
            A((-1, 1), (-1, 1)),
            StrideDirection.NW,
            [
                # stride 0
                P(1, 1),
                # stride 1
                P(0, 1),
                P(1, 0),
                # stride 2
                P(-1, 1),
                P(0, 0),
                P(1, -1),
                # stride 3
                P(-1, 0),
                P(0, -1),
                # stride 4
                P(-1, -1),
            ],
        ),
        # (
        #     A((-1, 1), (-2, 2)),
        #     'NW',
        #     [
        #     ],
        # ),
        (
            A((-1, 1), (-1, 1)),
            StrideDirection.NE,
            [
                # stride 0
                P(1, -1),
                # stride 1
                P(0, -1),
                P(1, 0),
                # stride 2
                P(-1, -1),
                P(0, 0),
                P(1, 1),
                # stride 3
                P(-1, 0),
                P(0, 1),
                # stride 4
                P(-1, 1),
            ],
        ),
        # (
        #     A((-1, 1), (-2, 2)),
        #     'NE',
        #     [
        #         P(0, 0),
        #         P(-1, 0),
        #         P(0, 1),
        #         P(-1, 1),
        #         P(0, 2),
        #         P(-1, 2),
        #     ],
        # ),
    ],
)
def test_diagonal_strides(
    area: Area, stride_direction: StrideDirection, expected: Sequence[Position]
):
    assert list(diagonal_strides(area, stride_direction)) == expected
