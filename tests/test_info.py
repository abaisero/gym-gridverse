import unittest

import numpy as np

from gym_gridverse.geometry import Area, DeltaPosition, Orientation, Position
from gym_gridverse.grid_object import Floor, Goal, Hidden, Wall
from gym_gridverse.info import Agent, Grid


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

    def test_object_types(self):
        grid = Grid(3, 4)

        self.assertSetEqual(grid.object_types(), set([Floor]))

        grid[Position(0, 0)] = Wall()
        self.assertSetEqual(grid.object_types(), set([Floor, Wall]))

        grid[Position(0, 0)] = Goal()
        self.assertSetEqual(grid.object_types(), set([Floor, Goal]))

        grid[Position(1, 1)] = Wall()
        self.assertSetEqual(grid.object_types(), set([Floor, Goal, Wall]))

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

    def test_draw_area(self):
        grid = Grid(3, 4)
        grid.draw_area(Area((0, 2), (0, 3)), object_factory=Wall)

        grid_expected = Grid.from_objects(
            [
                [Wall(), Wall(), Wall(), Wall()],
                [Wall(), Floor(), Floor(), Wall()],
                [Wall(), Wall(), Wall(), Wall()],
            ]
        )

        self.assertEqual(grid, grid_expected)

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

        subgrid = grid.subgrid(Area((-1, 3), (-1, 4)))
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

        subgrid = grid.subgrid(Area((1, 1), (1, 2)))
        subgrid_expected = Grid.from_objects([[Wall(), Floor()]])
        self.assertEqual(subgrid, subgrid_expected)

        subgrid = grid.subgrid(Area((-1, 1), (-1, 1)))
        subgrid_expected = Grid.from_objects(
            [
                [Hidden(), Hidden(), Hidden()],
                [Hidden(), Wall(), Floor()],
                [Hidden(), Floor(), Wall()],
            ]
        )
        self.assertEqual(subgrid, subgrid_expected)

        subgrid = grid.subgrid(Area((1, 3), (2, 4)))
        subgrid_expected = Grid.from_objects(
            [
                [Floor(), Wall(), Hidden()],
                [Wall(), Floor(), Hidden()],
                [Hidden(), Hidden(), Hidden()],
            ]
        )
        self.assertEqual(subgrid, subgrid_expected)

    def test_change_orientation(self):
        # checkerboard pattern
        grid = Grid.from_objects(
            [
                [Wall(), Floor(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall()],
                [Wall(), Floor(), Wall(), Floor()],
            ]
        )

        grid_N = grid.change_orientation(Orientation.N)
        grid_N_expected = Grid.from_objects(
            [
                [Wall(), Floor(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall()],
                [Wall(), Floor(), Wall(), Floor()],
            ]
        )
        self.assertEqual(grid_N, grid_N_expected)

        grid_S = grid.change_orientation(Orientation.S)
        grid_S_expected = Grid.from_objects(
            [
                [Floor(), Wall(), Floor(), Wall()],
                [Wall(), Floor(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall()],
            ]
        )
        self.assertEqual(grid_S, grid_S_expected)

        grid_E = grid.change_orientation(Orientation.E)
        grid_E_expected = Grid.from_objects(
            [
                [Floor(), Wall(), Floor()],
                [Wall(), Floor(), Wall()],
                [Floor(), Wall(), Floor()],
                [Wall(), Floor(), Wall()],
            ]
        )
        self.assertEqual(grid_E, grid_E_expected)

        grid_W = grid.change_orientation(Orientation.W)
        grid_W_expected = Grid.from_objects(
            [
                [Wall(), Floor(), Wall()],
                [Floor(), Wall(), Floor()],
                [Wall(), Floor(), Wall()],
                [Floor(), Wall(), Floor()],
            ]
        )
        self.assertEqual(grid_W, grid_W_expected)

    def test_as_array(self):
        # checkerboard pattern
        grid = Grid.from_objects(
            [
                [Wall(), Floor(), Wall(), Floor()],
                [Floor(), Wall(), Floor(), Wall()],
                [Wall(), Floor(), Wall(), Floor()],
            ]
        )
        array = grid.as_array()
        self.assertTupleEqual(array.shape, (3, 4, 3))
        np.testing.assert_array_equal(array[0, 0], array[0, 2])
        np.testing.assert_array_equal(array[0, 0], array[1, 1])
        np.testing.assert_array_equal(array[0, 0], array[1, 3])
        np.testing.assert_array_equal(array[0, 0], array[2, 0])
        np.testing.assert_array_equal(array[0, 0], array[2, 2])

        self.assertRaises(
            AssertionError,
            np.testing.assert_array_equal,
            array[0, 0],
            array[0, 1],
        )
        self.assertRaises(
            AssertionError,
            np.testing.assert_array_equal,
            array[0, 0],
            array[0, 3],
        )
        self.assertRaises(
            AssertionError,
            np.testing.assert_array_equal,
            array[0, 0],
            array[1, 0],
        )
        self.assertRaises(
            AssertionError,
            np.testing.assert_array_equal,
            array[0, 0],
            array[1, 2],
        )
        self.assertRaises(
            AssertionError,
            np.testing.assert_array_equal,
            array[0, 0],
            array[2, 1],
        )
        self.assertRaises(
            AssertionError,
            np.testing.assert_array_equal,
            array[0, 0],
            array[2, 3],
        )


class TestAgent(unittest.TestCase):
    def test_get_pov_area(self):
        relative_area = Area((-6, 0), (-3, 3))

        agent = Agent(Position(0, 0), Orientation.N)
        self.assertEqual(
            agent.get_pov_area(relative_area), Area((-6, 0), (-3, 3))
        )

        agent = Agent(Position(0, 0), Orientation.S)
        self.assertEqual(
            agent.get_pov_area(relative_area), Area((0, 6), (-3, 3))
        )

        agent = Agent(Position(0, 0), Orientation.E)
        self.assertEqual(
            agent.get_pov_area(relative_area), Area((-3, 3), (0, 6))
        )

        agent = Agent(Position(0, 0), Orientation.W)
        self.assertEqual(
            agent.get_pov_area(relative_area), Area((-3, 3), (-6, 0))
        )

        agent = Agent(Position(1, 2), Orientation.N)
        self.assertEqual(
            agent.get_pov_area(relative_area), Area((-5, 1), (-1, 5))
        )

        agent = Agent(Position(1, 2), Orientation.S)
        self.assertEqual(
            agent.get_pov_area(relative_area), Area((1, 7), (-1, 5))
        )

        agent = Agent(Position(1, 2), Orientation.E)
        self.assertEqual(
            agent.get_pov_area(relative_area), Area((-2, 4), (2, 8))
        )

        agent = Agent(Position(1, 2), Orientation.W)
        self.assertEqual(
            agent.get_pov_area(relative_area), Area((-2, 4), (-4, 2))
        )

    def test_position_relative(self):
        agent = Agent(Position(0, 0), Orientation.N)
        self.assertEqual(
            agent.position_relative(DeltaPosition(1, -1)), Position(1, -1)
        )

        agent = Agent(Position(0, 0), Orientation.S)
        self.assertEqual(
            agent.position_relative(DeltaPosition(1, -1)), Position(-1, 1)
        )

        agent = Agent(Position(0, 0), Orientation.E)
        self.assertEqual(
            agent.position_relative(DeltaPosition(1, -1)), Position(-1, -1)
        )

        agent = Agent(Position(0, 0), Orientation.W)
        self.assertEqual(
            agent.position_relative(DeltaPosition(1, -1)), Position(1, 1)
        )

        agent = Agent(Position(1, 2), Orientation.N)
        self.assertEqual(
            agent.position_relative(DeltaPosition(2, -2)), Position(3, 0)
        )

        agent = Agent(Position(1, 2), Orientation.S)
        self.assertEqual(
            agent.position_relative(DeltaPosition(2, -2)), Position(-1, 4)
        )

        agent = Agent(Position(1, 2), Orientation.E)
        self.assertEqual(
            agent.position_relative(DeltaPosition(2, -2)), Position(-1, 0)
        )

        agent = Agent(Position(1, 2), Orientation.W)
        self.assertEqual(
            agent.position_relative(DeltaPosition(2, -2)), Position(3, 4)
        )

    def test_position_in_front(self):
        agent = Agent(Position(0, 0), Orientation.N)
        self.assertEqual(agent.position_in_front(), Position(-1, 0))

        agent = Agent(Position(0, 0), Orientation.S)
        self.assertEqual(agent.position_in_front(), Position(1, 0))

        agent = Agent(Position(0, 0), Orientation.E)
        self.assertEqual(agent.position_in_front(), Position(0, 1))

        agent = Agent(Position(0, 0), Orientation.W)
        self.assertEqual(agent.position_in_front(), Position(0, -1))

        agent = Agent(Position(1, 2), Orientation.N)
        self.assertEqual(agent.position_in_front(), Position(0, 2))

        agent = Agent(Position(1, 2), Orientation.S)
        self.assertEqual(agent.position_in_front(), Position(2, 2))

        agent = Agent(Position(1, 2), Orientation.E)
        self.assertEqual(agent.position_in_front(), Position(1, 3))

        agent = Agent(Position(1, 2), Orientation.W)
        self.assertEqual(agent.position_in_front(), Position(1, 1))


if __name__ == '__main__':
    unittest.main()
