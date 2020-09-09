import random
import unittest

import gym

import gym_gridverse  # pylint: disable=unused-import


class TestGymEnvironment(unittest.TestCase):
    def test_registration(self):  # pylint: disable=no-self-use
        gym.make('GridVerse-MiniGrid-Empty-5x5-v0')

    def test_control_loop(self):  # pylint: disable=no-self-use
        env = gym.make('GridVerse-MiniGrid-Empty-5x5-v0')

        observation = env.reset()  # pylint: disable=unused-variable
        for _ in range(10):
            action = random.randrange(env.action_space.n)

            (
                observation,
                reward,  #  pylint: disable=unused-variable
                done,
                info,  #  pylint: disable=unused-variable
            ) = env.step(action)

            if done:
                observation = env.reset()


if __name__ == '__main__':
    unittest.main()
