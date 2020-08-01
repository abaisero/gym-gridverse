import unittest

from gym_gridverse.geometry import Area, Position
from gym_gridverse.grid_object import Floor, Hidden, Wall
from gym_gridverse.state import Grid


class TestGrid(unittest.TestCase):
    def test_shape(self):
        grid = Grid(3, 4)

        self.assertTupleEqual(grid.shape, (3, 4))

    def test_contains(self):
        grid = Grid(3, 4)

        self.assertIn(Position(0, 0), grid)
        self.assertIn(Position(2, 3), grid)

        self.assertNotIn(Position(-1, 0), grid)
        self.assertNotIn(Position(0, -1), grid)
        self.assertNotIn(Position(3, 3), grid)
        self.assertNotIn(Position(2, 4), grid)

    def test_check_contains(self):
        grid = Grid(3, 4)

        # pylint: disable=protected-access
        self.assertRaises(ValueError, grid._check_contains, Position(-1, 0))
        self.assertRaises(ValueError, grid._check_contains, Position(0, -1))
        self.assertRaises(ValueError, grid._check_contains, Position(3, 3))
        self.assertRaises(ValueError, grid._check_contains, Position(2, 4))

    def test_positions(self):
        grid = Grid(3, 4)

        positions = set(grid.positions())
        self.assertEqual(len(positions), 3 * 4)

        for position in positions:
            self.assertIn(position, grid)

    def test_get_position(self):
        grid = Grid(3, 4)

        # testing position -> grid_object -> position roundtrip
        for position in grid.positions():
            self.assertTupleEqual(grid.get_position(grid[position]), position)

        # testing exception when object is not in grid
        self.assertRaises(ValueError, grid.get_position, Floor())

    def test_get_item(self):
        grid = Grid(3, 4)

        pos = Position(0, 0)
        self.assertIsInstance(grid[pos], Floor)
        self.assertIs(grid[pos], grid[pos])

    def test_set_item(self):
        grid = Grid(3, 4)

        pos = Position(0, 0)
        obj = Floor()

        self.assertIsNot(grid[pos], obj)
        grid[pos] = obj
        self.assertIs(grid[pos], obj)

    def test_swap(self):
        grid = Grid(3, 4)

        # caching positions and objects before swap
        objects_before = {
            position: grid[position] for position in grid.positions()
        }

        pos1 = Position(0, 0)
        pos2 = Position(1, 1)
        grid.swap(pos1, pos2)

        # caching positions and objects after swap
        objects_after = {
            position: grid[position] for position in grid.positions()
        }

        # testing swapped objects
        self.assertIs(objects_before[pos1], objects_after[pos2])
        self.assertIs(objects_before[pos2], objects_after[pos1])

        # testing all other objects are the same
        for position in grid.positions():
            if position not in (pos1, pos2):
                self.assertIs(objects_before[position], objects_after[position])

    def test_subgrid(self):
        # checkerboard pattern
        grid = Grid.from_objects(
            [
                [Wall(), Floor(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall()],
                [Wall(), Floor(), Wall(), Floor()],
            ]
        )

        subgrid = grid.subgrid(Area(-1, -1, 3, 4))
        subgrid_expected = Grid.from_objects(
            [
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
                [Hidden(), Wall(), Floor(), Wall(), Floor(), Hidden()],
                [Hidden(), Floor(), Wall(), Floor(), Wall(), Hidden()],
                [Hidden(), Wall(), Floor(), Wall(), Floor(), Hidden()],
                [Hidden(), Hidden(), Hidden(), Hidden(), Hidden(), Hidden()],
            ]
        )
        self.assertEqual(subgrid, subgrid_expected)

        subgrid = grid.subgrid(Area(1, 1, 1, 2))
        subgrid_expected = Grid.from_objects([[Wall(), Floor()]])
        self.assertEqual(subgrid, subgrid_expected)

        subgrid = grid.subgrid(Area(-1, -1, 1, 1))
        subgrid_expected = Grid.from_objects(
            [
                [Hidden(), Hidden(), Hidden()],
                [Hidden(), Wall(), Floor()],
                [Hidden(), Floor(), Wall()],
            ]
        )
        self.assertEqual(subgrid, subgrid_expected)

        subgrid = grid.subgrid(Area(1, 2, 3, 4))
        subgrid_expected = Grid.from_objects(
            [
                [Floor(), Wall(), Hidden()],
                [Wall(), Floor(), Hidden()],
                [Hidden(), Hidden(), Hidden()],
            ]
        )
        self.assertEqual(subgrid, subgrid_expected)


if __name__ == '__main__':
    unittest.main()
