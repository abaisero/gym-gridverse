""" Tests Grid Object behavior and properties """
import unittest

import numpy as np
from gym_gridverse.grid_object import Colors, Floor


class TestFloor(unittest.TestCase):
    """ Basic stupid tests for floor grid object """

    def test_properties(self):
        """ Basic stupid tests for floor grid object """

        floor = Floor()

        self.assertTrue(floor.transparent)
        self.assertFalse(floor.blocks)
        self.assertEqual(floor.color, Colors.NONE)
        self.assertFalse(floor.can_be_picked_up)
        self.assertEqual(floor.state, 0)

        expected_arr_represtation = np.array([0, 0, 0])
        np.testing.assert_array_equal(
            floor.as_array(), expected_arr_represtation
        )


if __name__ == '__main__':
    unittest.main()
