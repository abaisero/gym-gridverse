import unittest

from gym_gridverse.geometry import Position
from gym_gridverse.grid_object import Floor
from gym_gridverse.state import Grid


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(3, 4)

    def test_shape(self):
        self.assertTupleEqual(self.grid.shape, (3, 4))

    def test_contains(self):
        self.assertIn(Position(0, 0), self.grid)
        self.assertIn(Position(2, 3), self.grid)

        self.assertNotIn(Position(-1, 0), self.grid)
        self.assertNotIn(Position(0, -1), self.grid)
        self.assertNotIn(Position(3, 3), self.grid)
        self.assertNotIn(Position(2, 4), self.grid)

    def test_check_contains(self):
        # pylint: disable=protected-access
        self.assertRaises(
            ValueError, self.grid._check_contains, Position(-1, 0)
        )
        self.assertRaises(
            ValueError, self.grid._check_contains, Position(0, -1)
        )
        self.assertRaises(ValueError, self.grid._check_contains, Position(3, 3))
        self.assertRaises(ValueError, self.grid._check_contains, Position(2, 4))

    def test_positions(self):
        positions = set(self.grid.positions())
        self.assertEqual(len(positions), 3 * 4)

        for position in positions:
            self.assertIn(position, self.grid)

    def test_get_position(self):
        # testing position -> grid_object -> position roundtrip
        for position in self.grid.positions():
            self.assertTupleEqual(
                self.grid.get_position(self.grid[position]), position
            )

        # testing exception when object is not in grid
        self.assertRaises(ValueError, self.grid.get_position, Floor())

    def test_get_item(self):
        pos = Position(0, 0)
        self.assertIs(self.grid[pos], self.grid[pos])

    def test_set_item(self):
        pos = Position(0, 0)
        obj = Floor()

        self.assertIsNot(self.grid[pos], obj)
        self.grid[pos] = obj
        self.assertIs(self.grid[pos], obj)

    def test_swap(self):
        # caching positions and objects before swap
        objects_before = {
            position: self.grid[position] for position in self.grid.positions()
        }

        pos1 = Position(0, 0)
        pos2 = Position(1, 1)
        self.grid.swap(pos1, pos2)

        # caching positions and objects after swap
        objects_after = {
            position: self.grid[position] for position in self.grid.positions()
        }

        # testing swapped objects
        self.assertIs(objects_before[pos1], objects_after[pos2])
        self.assertIs(objects_before[pos2], objects_after[pos1])

        # testing all other objects are the same
        for position in self.grid.positions():
            if position not in (pos1, pos2):
                self.assertIs(objects_before[position], objects_after[position])


if __name__ == '__main__':
    unittest.main()
