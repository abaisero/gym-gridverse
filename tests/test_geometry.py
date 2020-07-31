import unittest

from gym_gridverse.geometry import DeltaPosition, Orientation


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
