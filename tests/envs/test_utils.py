import random
import unittest

from gym_gridverse.actions import ROTATION_ACTIONS, Actions
from gym_gridverse.envs.utils import updated_agent_position_if_unobstructed
from gym_gridverse.geometry import Orientation, Position


class TestAgentMovement(unittest.TestCase):
    def test_unrelated_actoins(self):
        """ Any action that does not 'move' should not affect next position"""

        pos = Position(random.randint(0, 5), random.randint(0, 5))
        orientation = random.choice(list(Orientation))
        action = random.choice(ROTATION_ACTIONS)

        self.assertEqual(
            updated_agent_position_if_unobstructed(pos, orientation, action),
            pos,
        )

        self.assertEqual(
            updated_agent_position_if_unobstructed(
                pos, orientation, Actions.ACTUATE
            ),
            pos,
        )

        self.assertEqual(
            updated_agent_position_if_unobstructed(
                pos, orientation, Actions.PICK_N_DROP
            ),
            pos,
        )

    def test_basic_moves(self):

        self.assertEqual(
            updated_agent_position_if_unobstructed(
                Position(3, 6), Orientation.N, Actions.MOVE_FORWARD
            ),
            Position(2, 6),
        )

        self.assertEqual(
            updated_agent_position_if_unobstructed(
                Position(5, 2), Orientation.S, Actions.MOVE_FORWARD
            ),
            Position(6, 2),
        )

        self.assertEqual(
            updated_agent_position_if_unobstructed(
                Position(1, 2), Orientation.W, Actions.MOVE_BACKWARD
            ),
            Position(1, 3),
        )

        self.assertEqual(
            updated_agent_position_if_unobstructed(
                Position(4, 1), Orientation.E, Actions.MOVE_LEFT
            ),
            Position(3, 1),
        )

    def test_move_off_grid(self):

        self.assertEqual(
            updated_agent_position_if_unobstructed(
                Position(0, 1), Orientation.S, Actions.MOVE_BACKWARD
            ),
            Position(-1, 1),
        )

        self.assertEqual(
            updated_agent_position_if_unobstructed(
                Position(4, 0), Orientation.N, Actions.MOVE_LEFT
            ),
            Position(4, -1),
        )
