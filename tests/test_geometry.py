import math
from typing import Sequence

import pytest

from gym_gridverse.geometry import (
    Area,
    Orientation,
    Pose,
    Position,
    StrideDirection,
    diagonal_strides,
    get_manhattan_boundary,
)


def T(y: int, x: int, orientation: Orientation) -> Pose:
    return Pose(Position(y, x), orientation)


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
def test_area_translate(area: Area, position: Position, expected: Area):
    assert area.translate(position) == expected


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
def test_area_rotate(area: Area, orientation: Orientation, expected: Area):
    assert area.rotate(orientation) == expected


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
    assert pos1 + pos2 == expected


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
def test_position_subtract(pos1: Position, pos2: Position, expected: Position):
    assert pos1 - pos2 == expected


@pytest.mark.parametrize(
    'pos,expected',
    [(P(y, x), P(-y, -x)) for x in range(2) for y in range(2)],
)
def test_position_negative(pos: Position, expected: Position):
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
    'delta_position,orientation,expected',
    [
        # y basis
        (P(1, 0), O_.N, P(1, 0)),
        (P(1, 0), O_.S, P(-1, 0)),
        (P(1, 0), O_.E, P(0, -1)),
        (P(1, 0), O_.W, P(0, 1)),
        # x basis
        (P(0, 1), O_.N, P(0, 1)),
        (P(0, 1), O_.S, P(0, -1)),
        (P(0, 1), O_.E, P(1, 0)),
        (P(0, 1), O_.W, P(-1, 0)),
        # others
        (P(1, 2), O_.N, P(1, 2)),
        (P(1, 2), O_.S, P(-1, -2)),
        (P(1, 2), O_.E, P(2, -1)),
        (P(1, 2), O_.W, P(-2, 1)),
    ],
)
def test_delta_position_rotate_basis(
    delta_position: Position,
    orientation: Orientation,
    expected: Position,
):
    assert delta_position.rotate(orientation) == expected


@pytest.mark.parametrize(
    'pose,relative_position,expected',
    [
        # zero pose position and zero relative position
        (T(0, 0, O_.N), P(0, 0), P(0, 0)),
        (T(0, 0, O_.S), P(0, 0), P(0, 0)),
        (T(0, 0, O_.E), P(0, 0), P(0, 0)),
        (T(0, 0, O_.W), P(0, 0), P(0, 0)),
        # zero pose position and non-zero relative position
        (T(0, 0, O_.N), P(1, 1), P(1, 1)),
        (T(0, 0, O_.S), P(1, 1), P(-1, -1)),
        (T(0, 0, O_.E), P(1, 1), P(1, -1)),
        (T(0, 0, O_.W), P(1, 1), P(-1, 1)),
        # non-zero pose position and non-zero relative position
        (T(1, 2, O_.N), P(1, 1), P(2, 3)),
        (T(1, 2, O_.S), P(1, 1), P(0, 1)),
        (T(1, 2, O_.E), P(1, 1), P(2, 1)),
        (T(1, 2, O_.W), P(1, 1), P(0, 3)),
    ],
)
def test_pose_absolute_position(
    pose: Pose, relative_position: Position, expected: Position
):
    assert pose.absolute_position(relative_position) == expected


@pytest.mark.parametrize(
    'pose,expected',
    [
        # zero pose position
        (T(0, 0, O_.N), P(-1, 0)),
        (T(0, 0, O_.S), P(1, 0)),
        (T(0, 0, O_.E), P(0, 1)),
        (T(0, 0, O_.W), P(0, -1)),
        # non-zero pose position
        (T(1, 2, O_.N), P(0, 2)),
        (T(1, 2, O_.S), P(2, 2)),
        (T(1, 2, O_.E), P(1, 3)),
        (T(1, 2, O_.W), P(1, 1)),
    ],
)
def test_pose_front(pose: Pose, expected: Position):
    assert pose.front() == expected


@pytest.mark.parametrize(
    'pose,relative_area,expected',
    [
        # zero pose position and zero area
        (T(0, 0, O_.N), A((0, 0), (0, 0)), A((0, 0), (0, 0))),
        (T(0, 0, O_.S), A((0, 0), (0, 0)), A((0, 0), (0, 0))),
        (T(0, 0, O_.E), A((0, 0), (0, 0)), A((0, 0), (0, 0))),
        (T(0, 0, O_.W), A((0, 0), (0, 0)), A((0, 0), (0, 0))),
        # zero pose position and non-zero area
        (T(0, 0, O_.N), A((1, 2), (3, 4)), A((1, 2), (3, 4))),
        (T(0, 0, O_.S), A((1, 2), (3, 4)), A((-2, -1), (-4, -3))),
        (T(0, 0, O_.E), A((1, 2), (3, 4)), A((3, 4), (-2, -1))),
        (T(0, 0, O_.W), A((1, 2), (3, 4)), A((-4, -3), (1, 2))),
        # non-zero pose position and non-zero area
        (T(1, 2, O_.N), A((1, 2), (3, 4)), A((2, 3), (5, 6))),
        (T(1, 2, O_.S), A((1, 2), (3, 4)), A((-1, 0), (-2, -1))),
        (T(1, 2, O_.E), A((1, 2), (3, 4)), A((4, 5), (0, 1))),
        (T(1, 2, O_.W), A((1, 2), (3, 4)), A((-3, -2), (3, 4))),
    ],
)
def test_pose_absolute_area(pose: Pose, relative_area: Area, expected: Area):
    assert pose.absolute_area(relative_area) == expected


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
