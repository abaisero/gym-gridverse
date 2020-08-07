import math
import random
import unittest

from gym_gridverse.geometry import (
    DeltaPosition,
    Orientation,
    Position,
    get_manhattan_boundary,
)


class TestOrientation(unittest.TestCase):
    def test_as_delta_position(self):
        self.assertEqual(
            Orientation.N.as_delta_position(), DeltaPosition(-1, 0)
        )

        self.assertEqual(Orientation.S.as_delta_position(), DeltaPosition(1, 0))

        self.assertEqual(Orientation.E.as_delta_position(), DeltaPosition(0, 1))

        self.assertEqual(
            Orientation.W.as_delta_position(), DeltaPosition(0, -1)
        )

    def test_as_delta_position_with_dist(self):
        self.assertEqual(
            Orientation.N.as_delta_position(2), DeltaPosition(-2, 0)
        )

        self.assertEqual(
            Orientation.S.as_delta_position(2), DeltaPosition(2, 0)
        )

        self.assertEqual(
            Orientation.E.as_delta_position(2), DeltaPosition(0, 2)
        )

        self.assertEqual(
            Orientation.W.as_delta_position(2), DeltaPosition(0, -2)
        )


class TestNeighborhoods(unittest.TestCase):
    """ Manhattan, box, etc"""

    def test_manhattan_boundary(self):
        """`get_manhattan_boundary`"""

        manhat_boundary = get_manhattan_boundary(Position(2, 2), 1)
        self.assertEqual(len(manhat_boundary), 4)
        self.assertIn(Position(1, 2), manhat_boundary)
        self.assertIn(Position(2, 3), manhat_boundary)
        self.assertIn(Position(3, 2), manhat_boundary)
        self.assertIn(Position(2, 1), manhat_boundary)

        manhat_boundary = get_manhattan_boundary(Position(4, 3), 2)
        self.assertEqual(len(manhat_boundary), 8)
        self.assertIn(Position(2, 3), manhat_boundary)
        self.assertIn(Position(3, 4), manhat_boundary)
        self.assertIn(Position(4, 5), manhat_boundary)
        self.assertIn(Position(5, 4), manhat_boundary)
        self.assertIn(Position(6, 3), manhat_boundary)
        self.assertIn(Position(5, 2), manhat_boundary)
        self.assertIn(Position(4, 1), manhat_boundary)
        self.assertIn(Position(3, 2), manhat_boundary)


class TestPosition(unittest.TestCase):
    def test_add(self):
        x1 = random.randint(-5, 10)
        x2 = random.randint(-5, 10)
        y1 = random.randint(-5, 10)
        y2 = random.randint(-5, 10)

        p1 = Position(x1, y1)
        p2 = Position(x2, y2)

        x3 = x1 + x2
        y3 = y1 + y2

        self.assertEqual(Position(x3, y3), Position.add(p1, p2))

    def test_subtract(self):
        x1 = random.randint(-5, 10)
        x2 = random.randint(-5, 10)
        y1 = random.randint(-5, 10)
        y2 = random.randint(-5, 10)

        p1 = Position(x1, y1)
        p2 = Position(x2, y2)

        x3 = x1 - x2
        y3 = y1 - y2

        self.assertEqual(Position(x3, y3), Position.subtract(p1, p2))

    def test_manhattan_distance(self):
        self.assertEqual(
            Position.manhattan_distance(Position(0, 0), Position(0, 0)), 0.0
        )
        self.assertEqual(
            Position.manhattan_distance(Position(0, 0), Position(0, 1)), 1.0
        )
        self.assertEqual(
            Position.manhattan_distance(Position(0, 0), Position(1, 1)), 2.0
        )
        self.assertEqual(
            Position.manhattan_distance(Position(0, 1), Position(1, 1)), 1.0
        )
        self.assertEqual(
            Position.manhattan_distance(Position(1, 1), Position(1, 1)), 0.0
        )

        # diagonal
        self.assertEqual(
            Position.manhattan_distance(Position(0, 0), Position(0, 0)), 0.0
        )
        self.assertEqual(
            Position.manhattan_distance(Position(0, 0), Position(1, 1)), 2.0
        )
        self.assertEqual(
            Position.manhattan_distance(Position(0, 0), Position(2, 2)), 4.0
        )
        self.assertEqual(
            Position.manhattan_distance(Position(0, 0), Position(3, 3)), 6.0
        )

    def test_euclidean_distance(self):
        self.assertEqual(
            Position.euclidean_distance(Position(0, 0), Position(0, 0)), 0.0
        )
        self.assertEqual(
            Position.euclidean_distance(Position(0, 0), Position(0, 1)), 1.0
        )
        self.assertEqual(
            Position.euclidean_distance(Position(0, 0), Position(1, 1)),
            math.sqrt(2.0),
        )
        self.assertEqual(
            Position.euclidean_distance(Position(0, 1), Position(1, 1)), 1.0
        )
        self.assertEqual(
            Position.euclidean_distance(Position(1, 1), Position(1, 1)), 0.0
        )

        # diagonal
        self.assertEqual(
            Position.euclidean_distance(Position(0, 0), Position(0, 0)), 0.0
        )
        self.assertEqual(
            Position.euclidean_distance(Position(0, 0), Position(1, 1)),
            math.sqrt(2.0),
        )
        self.assertEqual(
            Position.euclidean_distance(Position(0, 0), Position(2, 2)),
            math.sqrt(8.0),
        )
        self.assertEqual(
            Position.euclidean_distance(Position(0, 0), Position(3, 3)),
            math.sqrt(18.0),
        )


class TestDeltaPosition(unittest.TestCase):
    def test_rotate_basis(self):
        # y basis
        self.assertEqual(
            DeltaPosition(1, 0).rotate(Orientation.N), DeltaPosition(1, 0)
        )

        self.assertEqual(
            DeltaPosition(1, 0).rotate(Orientation.S), DeltaPosition(-1, 0)
        )

        self.assertEqual(
            DeltaPosition(1, 0).rotate(Orientation.E), DeltaPosition(0, -1)
        )

        self.assertEqual(
            DeltaPosition(1, 0).rotate(Orientation.W), DeltaPosition(0, 1)
        )

        # x basis
        self.assertEqual(
            DeltaPosition(0, 1).rotate(Orientation.N), DeltaPosition(0, 1)
        )

        self.assertEqual(
            DeltaPosition(0, 1).rotate(Orientation.S), DeltaPosition(0, -1)
        )

        self.assertEqual(
            DeltaPosition(0, 1).rotate(Orientation.E), DeltaPosition(1, 0)
        )

        self.assertEqual(
            DeltaPosition(0, 1).rotate(Orientation.W), DeltaPosition(-1, 0)
        )

    def test_rotate(self):
        self.assertEqual(
            DeltaPosition(1, 2).rotate(Orientation.N), DeltaPosition(1, 2)
        )

        self.assertEqual(
            DeltaPosition(1, 2).rotate(Orientation.S), DeltaPosition(-1, -2)
        )

        self.assertEqual(
            DeltaPosition(1, 2).rotate(Orientation.E), DeltaPosition(2, -1)
        )

        self.assertEqual(
            DeltaPosition(1, 2).rotate(Orientation.W), DeltaPosition(-2, 1)
        )
