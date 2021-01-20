from typing import Optional

import gym
import numpy as np
import pytest

from gym_gridverse.gym import GymStateWrapper


@pytest.mark.parametrize(
    'env_id',
    [
        'GridVerse-Empty-5x5-v0',
        'GridVerse-Empty-Random-5x5-v0',
        'GridVerse-Empty-6x6-v0',
        'GridVerse-Empty-Random-6x6-v0',
        'GridVerse-Empty-8x8-v0',
        'GridVerse-Empty-16x16-v0',
        'GridVerse-FourRooms-v0',
        'GridVerse-Dynamic-Obstacles-5x5-v0',
        'GridVerse-Dynamic-Obstacles-Random-5x5-v0',
        'GridVerse-Dynamic-Obstacles-6x6-v0',
        'GridVerse-Dynamic-Obstacles-Random-6x6-v0',
        'GridVerse-Dynamic-Obstacles-8x8-v0',
        'GridVerse-Dynamic-Obstacles-16x16-v0',
        'GridVerse-KeyDoor-5x5-v0',
        'GridVerse-KeyDoor-6x6-v0',
        'GridVerse-KeyDoor-8x8-v0',
        'GridVerse-KeyDoor-16x16-v0',
    ],
)
def test_gym_registration(env_id: str):
    gym.make(env_id)


@pytest.mark.parametrize(
    'env_id',
    [
        'GridVerse-Empty-Random-5x5-v0',
        'GridVerse-Empty-Random-6x6-v0',
        'GridVerse-Dynamic-Obstacles-Random-5x5-v0',
        'GridVerse-Dynamic-Obstacles-Random-6x6-v0',
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
    env = gym.make('GridVerse-Empty-5x5-v0')

    env.reset()  # pylint: disable=unused-variable
    for _ in range(10):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

        if done:
            env.reset()


@pytest.mark.parametrize(
    'env_id,representation',
    [
        ('GridVerse-Empty-5x5-v0', 'default'),
        ('GridVerse-Empty-5x5-v0', 'no_overlap'),
        ('GridVerse-KeyDoor-16x16-v0', 'default'),
        ('GridVerse-KeyDoor-16x16-v0', 'no_overlap'),
    ]
)
def test_gym_state_wrapper(env_id: str, representation: str):
    env = gym.make(env_id)
    with pytest.raises(AssertionError):
        wrapped_env = GymStateWrapper(env)
    env.set_state_representation(representation)
    o_space = env.observation_space
    wrapped_env = GymStateWrapper(env)
    so_space = wrapped_env.observation_space
    so = wrapped_env.reset()
    np.testing.assert_equal(so, wrapped_env.state)
    np.testing.assert_equal(so_space, env.state_space)
    np.testing.assert_equal(o_space, env.unwrapped.observation_space)
    for _ in range(10):
        action = wrapped_env.action_space.sample()
        o, _, done, _ = wrapped_env.step(action)
        np.testing.assert_equal(o, wrapped_env.state)
        if done:
            env.reset()

