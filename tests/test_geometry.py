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
        (A((0, 1), (0, 2)), O_.FORWARD, A((0, 1), (0, 2))),
        (A((0, 1), (0, 2)), O_.BACKWARD, A((-1, 0), (-2, 0))),
        (A((0, 1), (0, 2)), O_.RIGHT, A((0, 2), (-1, 0))),
        (A((0, 1), (0, 2)), O_.LEFT, A((-2, 0), (0, 1))),
        #
        (A((-1, 1), (-2, 2)), O_.FORWARD, A((-1, 1), (-2, 2))),
        (A((-1, 1), (-2, 2)), O_.BACKWARD, A((-1, 1), (-2, 2))),
        (A((-1, 1), (-2, 2)), O_.RIGHT, A((-2, 2), (-1, 1))),
        (A((-1, 1), (-2, 2)), O_.LEFT, A((-2, 2), (-1, 1))),
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
    'orientation,expected',
    [
        (O_.FORWARD, P(-1, 0)),
        (O_.BACKWARD, P(1, 0)),
        (O_.RIGHT, P(0, 1)),
        (O_.LEFT, P(0, -1)),
    ],
)
def test_position_from_orientation(
    orientation: Orientation, expected: Position
):
    assert Position.from_orientation(orientation) == expected


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


@pytest.mark.parametrize('py', range(-1, 1))
@pytest.mark.parametrize('px', range(-1, 1))
@pytest.mark.parametrize('qy', range(-1, 1))
@pytest.mark.parametrize('qx', range(-1, 1))
def test_position_add(py: int, px: int, qy: int, qx: int):
    p = Position(py, px)
    q = Position(qy, qx)
    expected = Position(py + qy, px + qx)
    assert p + q == q + p == expected


@pytest.mark.parametrize('py', [-1, 1])
@pytest.mark.parametrize('px', [-1, 1])
@pytest.mark.parametrize('qy', [-1, 1])
@pytest.mark.parametrize('qx', [-1, 1])
def test_position_sub(py: int, px: int, qy: int, qx: int):
    p = Position(py, px)
    q = Position(qy, qx)
    expected = Position(py - qy, px - qx)
    assert p - q == expected


@pytest.mark.parametrize('y', [-1, 1])
@pytest.mark.parametrize('x', [-1, 1])
def test_position_neg(y: int, x: int):
    position = Position(y, x)
    expected = Position(-y, -x)
    assert -position == expected


@pytest.mark.parametrize(
    'p,q,expected',
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
def test_position_manhattan_distance(p: Position, q: Position, expected: float):
    assert Position.manhattan_distance(p, q) == expected


@pytest.mark.parametrize(
    'p,q,expected',
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
def test_position_euclidean_distance(p: Position, q: Position, expected: float):
    assert Position.euclidean_distance(p, q) == expected


@pytest.mark.parametrize(
    'orientation,position,expected',
    [
        # y basis
        (O_.FORWARD, P(1, 0), P(1, 0)),
        (O_.BACKWARD, P(1, 0), P(-1, 0)),
        (O_.RIGHT, P(1, 0), P(0, -1)),
        (O_.LEFT, P(1, 0), P(0, 1)),
        #       x basis
        (O_.FORWARD, P(0, 1), P(0, 1)),
        (O_.BACKWARD, P(0, 1), P(0, -1)),
        (O_.RIGHT, P(0, 1), P(1, 0)),
        (O_.LEFT, P(0, 1), P(-1, 0)),
        #       others
        (O_.FORWARD, P(1, 2), P(1, 2)),
        (O_.BACKWARD, P(1, 2), P(-1, -2)),
        (O_.RIGHT, P(1, 2), P(2, -1)),
        (O_.LEFT, P(1, 2), P(-2, 1)),
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
        (T(1, 1, O_.RIGHT), P(1, 1), P(2, 0)),
        (T(1, 1, O_.RIGHT), P(1, -1), P(0, 0)),
        (T(1, 1, O_.RIGHT), P(-1, 1), P(2, 2)),
        (T(1, 1, O_.RIGHT), P(-1, -1), P(0, 2)),
    ],
)
def test_transform_mul_position(
    transform: Transform, position: Position, expected: Position
):
    assert transform * position == position * transform == expected


@pytest.mark.parametrize(
    'transform,orientation,expected',
    [
        (T(1, 1, O_.RIGHT), O_.FORWARD, O_.RIGHT),
        (T(1, 1, O_.RIGHT), O_.RIGHT, O_.BACKWARD),
        (T(1, 1, O_.RIGHT), O_.BACKWARD, O_.LEFT),
        (T(1, 1, O_.RIGHT), O_.LEFT, O_.FORWARD),
    ],
)
def test_transform_mul_orientation(
    transform: Transform, orientation: Orientation, expected: Orientation
):
    assert transform * orientation == orientation * transform == expected


@pytest.mark.parametrize(
    't,s,expected',
    [
        (T(0, 0, O_.FORWARD), T(0, 0, O_.FORWARD), T(0, 0, O_.FORWARD)),
        (T(0, 0, O_.FORWARD), T(-1, 1, O_.RIGHT), T(-1, 1, O_.RIGHT)),
        (T(0, 0, O_.FORWARD), T(-2, 2, O_.BACKWARD), T(-2, 2, O_.BACKWARD)),
        (T(0, 0, O_.FORWARD), T(-3, 3, O_.LEFT), T(-3, 3, O_.LEFT)),
        #
        (T(-1, 1, O_.RIGHT), T(0, 0, O_.FORWARD), T(-1, 1, O_.RIGHT)),
        (T(-1, 1, O_.RIGHT), T(-1, 1, O_.RIGHT), T(0, 2, O_.BACKWARD)),
        (T(-1, 1, O_.RIGHT), T(-2, 2, O_.BACKWARD), T(1, 3, O_.LEFT)),
        (T(-1, 1, O_.RIGHT), T(-3, 3, O_.LEFT), T(2, 4, O_.FORWARD)),
        #
        (T(-2, 2, O_.BACKWARD), T(0, 0, O_.FORWARD), T(-2, 2, O_.BACKWARD)),
        (T(-2, 2, O_.BACKWARD), T(-1, 1, O_.RIGHT), T(-1, 1, O_.LEFT)),
        (T(-2, 2, O_.BACKWARD), T(-2, 2, O_.BACKWARD), T(0, 0, O_.FORWARD)),
        (T(-2, 2, O_.BACKWARD), T(-3, 3, O_.LEFT), T(1, -1, O_.RIGHT)),
        #
        (T(-3, 3, O_.LEFT), T(0, 0, O_.FORWARD), T(-3, 3, O_.LEFT)),
        (T(-3, 3, O_.LEFT), T(-1, 1, O_.RIGHT), T(-4, 2, O_.FORWARD)),
        (T(-3, 3, O_.LEFT), T(-2, 2, O_.BACKWARD), T(-5, 1, O_.RIGHT)),
        (T(-3, 3, O_.LEFT), T(-3, 3, O_.LEFT), T(-6, 0, O_.BACKWARD)),
    ],
)
def test_transform_mul_transform(
    t: Transform, s: Transform, expected: Transform
):
    assert t * s == expected


@pytest.mark.parametrize(
    'transform,expected',
    [
        (T(0, 0, O_.FORWARD), T(0, 0, O_.FORWARD)),
        (T(-1, 1, O_.RIGHT), T(1, 1, O_.LEFT)),
        (T(-2, 2, O_.BACKWARD), T(-2, 2, O_.BACKWARD)),
        (T(-3, 3, O_.LEFT), T(-3, -3, O_.RIGHT)),
    ],
)
def test_transform_neg(transform: Transform, expected: Transform):
    assert (-transform) == expected


@pytest.mark.parametrize(
    'transform',
    [
        T(0, 0, O_.FORWARD),
        T(-1, 1, O_.RIGHT),
        T(-2, 2, O_.BACKWARD),
        T(-3, 3, O_.LEFT),
    ],
)
def test_transform_neg_identity(transform: Transform):
    assert (
        transform * (-transform)
        == -transform * transform
        == T(0, 0, O_.FORWARD)
    )


@pytest.mark.parametrize(
    'transform,area,expected',
    [
        (T(0, 0, O_.FORWARD), Area((1, 2), (1, 2)), Area((1, 2), (1, 2))),
        (T(0, 0, O_.FORWARD), Area((1, 2), (2, 4)), Area((1, 2), (2, 4))),
        (T(0, 0, O_.FORWARD), Area((2, 4), (1, 2)), Area((2, 4), (1, 2))),
        (T(0, 0, O_.FORWARD), Area((2, 4), (2, 4)), Area((2, 4), (2, 4))),
        #
        (T(-1, 1, O_.RIGHT), Area((1, 2), (1, 2)), Area((0, 1), (-1, 0))),
        (T(-1, 1, O_.RIGHT), Area((1, 2), (2, 4)), Area((1, 3), (-1, 0))),
        (T(-1, 1, O_.RIGHT), Area((2, 4), (1, 2)), Area((0, 1), (-3, -1))),
        (T(-1, 1, O_.RIGHT), Area((2, 4), (2, 4)), Area((1, 3), (-3, -1))),
        #
        (T(-2, 2, O_.BACKWARD), Area((1, 2), (1, 2)), Area((-4, -3), (0, 1))),
        (T(-2, 2, O_.BACKWARD), Area((1, 2), (2, 4)), Area((-4, -3), (-2, 0))),
        (T(-2, 2, O_.BACKWARD), Area((2, 4), (1, 2)), Area((-6, -4), (0, 1))),
        (T(-2, 2, O_.BACKWARD), Area((2, 4), (2, 4)), Area((-6, -4), (-2, 0))),
        #
        (T(-3, 3, O_.LEFT), Area((1, 2), (1, 2)), Area((-5, -4), (4, 5))),
        (T(-3, 3, O_.LEFT), Area((1, 2), (2, 4)), Area((-7, -5), (4, 5))),
        (T(-3, 3, O_.LEFT), Area((2, 4), (1, 2)), Area((-5, -4), (5, 7))),
        (T(-3, 3, O_.LEFT), Area((2, 4), (2, 4)), Area((-7, -5), (5, 7))),
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
