import numpy as np
import unittest

from gym_gridverse.actions import Actions
from gym_gridverse.data_gen import sample_transitions
from gym_gridverse.envs.factory import gym_minigrid_from_descr
from gym_gridverse.representations.observation_representations import (
    DefaultObservationRepresentation,
)
from gym_gridverse.representations.state_representations import (
    DefaultStateRepresentation,
)
from gym_gridverse.simulator import Simulator


class TestSampleTransitions(unittest.TestCase):
    def test_that_it_runs(self):

        domain = gym_minigrid_from_descr(
            "MiniGrid-Dynamic-Obstacles-Random-6x6-v0"
        )

        s_rep = DefaultStateRepresentation(domain.state_space)
        o_rep = DefaultObservationRepresentation(domain.observation_space)

        sim = Simulator(domain, s_rep, o_rep)

        def action_sampler(_s) -> Actions:
            return np.random.choice(Actions)

        num_samples = 3
        state_sampler = domain.functional_reset
        sample_transitions(num_samples, state_sampler, action_sampler, sim)

        self.assertTrue(True)  # pylint: disable=redundant-unittest-assert


if __name__ == '__main__':
    unittest.main()
