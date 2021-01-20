from typing import Optional

import gym
import numpy as np
import pytest


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
@pytest.mark.parametrize(
    'obs_rep_override',
    [
        None,
        'default',
        'no_overlap'
    ]
)
@pytest.mark.parametrize(
    'state_rep_override',
    [
        None,
        'default',
        'no_overlap'
    ]
)
def test_gym_registration(env_id: str, obs_rep_override: [str, None], state_rep_override: [str, None]):
    e0 = gym.make(env_id)
    e = gym.make(env_id, obs_rep_override=obs_rep_override, state_rep_override=state_rep_override)
    assert state_rep_override is None or e.state_space is not None
    if obs_rep_override not in (None, 'default'): e0.set_observation_representation(obs_rep_override)
    np.testing.assert_equal(e.observation_space, e0.observation_space)


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
