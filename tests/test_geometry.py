import math
from typing import Sequence

import pytest

from gym_gridverse.geometry import (
    Area,
    Orientation,
    Position,
    Transform,
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
        (A((0, 1), (0, 2)), O_.F, A((0, 1), (0, 2))),
        (A((0, 1), (0, 2)), O_.B, A((-1, 0), (-2, 0))),
        (A((0, 1), (0, 2)), O_.R, A((0, 2), (-1, 0))),
        (A((0, 1), (0, 2)), O_.L, A((-2, 0), (0, 1))),
        #
        (A((-1, 1), (-2, 2)), O_.F, A((-1, 1), (-2, 2))),
        (A((-1, 1), (-2, 2)), O_.B, A((-1, 1), (-2, 2))),
        (A((-1, 1), (-2, 2)), O_.R, A((-2, 2), (-1, 1))),
        (A((-1, 1), (-2, 2)), O_.L, A((-2, 2), (-1, 1))),
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
        (O_.F, P(-1, 0)),
        (O_.B, P(1, 0)),
        (O_.R, P(0, 1)),
        (O_.L, P(0, -1)),
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
        (O_.F, P(1, 0), P(1, 0)),
        (O_.B, P(1, 0), P(-1, 0)),
        (O_.R, P(1, 0), P(0, -1)),
        (O_.L, P(1, 0), P(0, 1)),
        #       x basis
        (O_.F, P(0, 1), P(0, 1)),
        (O_.B, P(0, 1), P(0, -1)),
        (O_.R, P(0, 1), P(1, 0)),
        (O_.L, P(0, 1), P(-1, 0)),
        #       others
        (O_.F, P(1, 2), P(1, 2)),
        (O_.B, P(1, 2), P(-1, -2)),
        (O_.R, P(1, 2), P(2, -1)),
        (O_.L, P(1, 2), P(-2, 1)),
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
        (T(1, 1, O_.R), P(1, 1), P(2, 0)),
        (T(1, 1, O_.R), P(1, -1), P(0, 0)),
        (T(1, 1, O_.R), P(-1, 1), P(2, 2)),
        (T(1, 1, O_.R), P(-1, -1), P(0, 2)),
    ],
)
def test_transform_mul_position(
    transform: Transform, position: Position, expected: Position
):
    assert transform * position == position * transform == expected


@pytest.mark.parametrize(
    'transform,orientation,expected',
    [
        (T(1, 1, O_.R), O_.F, O_.R),
        (T(1, 1, O_.R), O_.R, O_.B),
        (T(1, 1, O_.R), O_.B, O_.L),
        (T(1, 1, O_.R), O_.L, O_.F),
    ],
)
def test_transform_mul_orientation(
    transform: Transform, orientation: Orientation, expected: Orientation
):
    assert transform * orientation == orientation * transform == expected


@pytest.mark.parametrize(
    't,s,expected',
    [
        (T(0, 0, O_.F), T(0, 0, O_.F), T(0, 0, O_.F)),
        (T(0, 0, O_.F), T(-1, 1, O_.R), T(-1, 1, O_.R)),
        (T(0, 0, O_.F), T(-2, 2, O_.B), T(-2, 2, O_.B)),
        (T(0, 0, O_.F), T(-3, 3, O_.L), T(-3, 3, O_.L)),
        #
        (T(-1, 1, O_.R), T(0, 0, O_.F), T(-1, 1, O_.R)),
        (T(-1, 1, O_.R), T(-1, 1, O_.R), T(0, 2, O_.B)),
        (T(-1, 1, O_.R), T(-2, 2, O_.B), T(1, 3, O_.L)),
        (T(-1, 1, O_.R), T(-3, 3, O_.L), T(2, 4, O_.F)),
        #
        (T(-2, 2, O_.B), T(0, 0, O_.F), T(-2, 2, O_.B)),
        (T(-2, 2, O_.B), T(-1, 1, O_.R), T(-1, 1, O_.L)),
        (T(-2, 2, O_.B), T(-2, 2, O_.B), T(0, 0, O_.F)),
        (T(-2, 2, O_.B), T(-3, 3, O_.L), T(1, -1, O_.R)),
        #
        (T(-3, 3, O_.L), T(0, 0, O_.F), T(-3, 3, O_.L)),
        (T(-3, 3, O_.L), T(-1, 1, O_.R), T(-4, 2, O_.F)),
        (T(-3, 3, O_.L), T(-2, 2, O_.B), T(-5, 1, O_.R)),
        (T(-3, 3, O_.L), T(-3, 3, O_.L), T(-6, 0, O_.B)),
    ],
)
def test_transform_mul_transform(
    t: Transform, s: Transform, expected: Transform
):
    assert t * s == expected


@pytest.mark.parametrize(
    'transform,expected',
    [
        (T(0, 0, O_.F), T(0, 0, O_.F)),
        (T(-1, 1, O_.R), T(1, 1, O_.L)),
        (T(-2, 2, O_.B), T(-2, 2, O_.B)),
        (T(-3, 3, O_.L), T(-3, -3, O_.R)),
    ],
)
def test_transform_neg(transform: Transform, expected: Transform):
    assert (-transform) == expected


@pytest.mark.parametrize(
    'transform',
    [
        T(0, 0, O_.F),
        T(-1, 1, O_.R),
        T(-2, 2, O_.B),
        T(-3, 3, O_.L),
    ],
)
def test_transform_neg_identity(transform: Transform):
    assert transform * (-transform) == -transform * transform == T(0, 0, O_.F)


@pytest.mark.parametrize(
    'transform,area,expected',
    [
        (T(0, 0, O_.F), Area((1, 2), (1, 2)), Area((1, 2), (1, 2))),
        (T(0, 0, O_.F), Area((1, 2), (2, 4)), Area((1, 2), (2, 4))),
        (T(0, 0, O_.F), Area((2, 4), (1, 2)), Area((2, 4), (1, 2))),
        (T(0, 0, O_.F), Area((2, 4), (2, 4)), Area((2, 4), (2, 4))),
        #
        (T(-1, 1, O_.R), Area((1, 2), (1, 2)), Area((0, 1), (-1, 0))),
        (T(-1, 1, O_.R), Area((1, 2), (2, 4)), Area((1, 3), (-1, 0))),
        (T(-1, 1, O_.R), Area((2, 4), (1, 2)), Area((0, 1), (-3, -1))),
        (T(-1, 1, O_.R), Area((2, 4), (2, 4)), Area((1, 3), (-3, -1))),
        #
        (T(-2, 2, O_.B), Area((1, 2), (1, 2)), Area((-4, -3), (0, 1))),
        (T(-2, 2, O_.B), Area((1, 2), (2, 4)), Area((-4, -3), (-2, 0))),
        (T(-2, 2, O_.B), Area((2, 4), (1, 2)), Area((-6, -4), (0, 1))),
        (T(-2, 2, O_.B), Area((2, 4), (2, 4)), Area((-6, -4), (-2, 0))),
        #
        (T(-3, 3, O_.L), Area((1, 2), (1, 2)), Area((-5, -4), (4, 5))),
        (T(-3, 3, O_.L), Area((1, 2), (2, 4)), Area((-7, -5), (4, 5))),
        (T(-3, 3, O_.L), Area((2, 4), (1, 2)), Area((-5, -4), (5, 7))),
        (T(-3, 3, O_.L), Area((2, 4), (2, 4)), Area((-7, -5), (5, 7))),
    ],
)
def test_transform_mul_area(transform: Transform, area: Area, expected: Area):
    assert transform * area == expected
