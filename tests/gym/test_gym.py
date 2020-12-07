from typing import Optional

import gym
import numpy as np
import pytest


@pytest.mark.parametrize(
    'env_id',
    [
        'GridVerse-MiniGrid-Empty-5x5-v0',
        'GridVerse-MiniGrid-Empty-Random-5x5-v0',
        'GridVerse-MiniGrid-Empty-6x6-v0',
        'GridVerse-MiniGrid-Empty-Random-6x6-v0',
        'GridVerse-MiniGrid-Empty-8x8-v0',
        'GridVerse-MiniGrid-Empty-16x16-v0',
        'GridVerse-MiniGrid-FourRooms-v0',
        'GridVerse-MiniGrid-Dynamic-Obstacles-5x5-v0',
        'GridVerse-MiniGrid-Dynamic-Obstacles-Random-5x5-v0',
        'GridVerse-MiniGrid-Dynamic-Obstacles-6x6-v0',
        'GridVerse-MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
        'GridVerse-MiniGrid-Dynamic-Obstacles-8x8-v0',
        'GridVerse-MiniGrid-Dynamic-Obstacles-16x16-v0',
        'GridVerse-MiniGrid-KeyDoor-5x5-v0',
        'GridVerse-MiniGrid-KeyDoor-6x6-v0',
        'GridVerse-MiniGrid-KeyDoor-8x8-v0',
        'GridVerse-MiniGrid-KeyDoor-16x16-v0',
    ],
)
def test_gym_registration(env_id: str):
    gym.make(env_id)


@pytest.mark.parametrize(
    'env_id',
    [
        'GridVerse-MiniGrid-Empty-Random-5x5-v0',
        'GridVerse-MiniGrid-Empty-Random-6x6-v0',
        'GridVerse-MiniGrid-Dynamic-Obstacles-Random-5x5-v0',
        'GridVerse-MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
    ],
)
@pytest.mark.parametrize('seed', [1, 10, 1337, 0xDEADBEEF])
def test_gym_seed(env_id: str, seed: Optional[int]):
    env = gym.make(env_id)

    env.seed(seed)
    observation1 = env.reset()

    env.seed(seed)
    observation2 = env.reset()

    np.testing.assert_equal(observation1, observation2)


def test_gym_control_loop():
    env = gym.make('GridVerse-MiniGrid-Empty-5x5-v0')

    env.reset()  # pylint: disable=unused-variable
    for _ in range(10):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

        if done:
            env.reset()
