from typing import Optional

import gym
import numpy as np
import pytest

from gym_gridverse.gym import GymStateWrapper


@pytest.mark.parametrize(
    'env_id',
    [
        "GV-Crossing-5x5-v0",
        "GV-Crossing-7x7-v0",
        "GV-DynamicObstacles-5x5-v0",
        "GV-DynamicObstacles-7x7-v0",
        "GV-Empty-4x4-v0",
        "GV-Empty-8x8-v0",
        "GV-FourRooms-7x7-v0",
        "GV-FourRooms-9x9-v0",
        "GV-Keydoor-5x5-v0",
        "GV-Keydoor-7x7-v0",
        "GV-Keydoor-9x9-v0",
        "GV-Memory-5x5-v0",
        "GV-Memory-9x9-v0",
        "GV-MemoryFourRooms-7x7-v0",
        "GV-MemoryFourRooms-9x9-v0",
        "GV-MemoryNineRooms-10x10-v0",
        "GV-MemoryNineRooms-13x13-v0",
        "GV-NineRooms-10x10-v0",
        "GV-NineRooms-13x13-v0",
        "GV-Teleport-5x5-v0",
        "GV-Teleport-7x7-v0",
    ],
)
def test_gym_registration(env_id: str):
    gym.make(env_id)


@pytest.mark.parametrize(
    'env_id',
    [
        "GV-Empty-4x4-v0",
        "GV-Empty-8x8-v0",
        "GV-DynamicObstacles-5x5-v0",
        "GV-DynamicObstacles-7x7-v0",
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
    env = gym.make('GV-Empty-4x4-v0')

    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

        if done:
            env.reset()


@pytest.mark.parametrize('env_id', ['GV-Empty-4x4-v0', 'GV-Keydoor-9x9-v0'])
@pytest.mark.parametrize('representation', ['default', 'no-overlap'])
def test_gym_state_wrapper(env_id: str, representation: str):
    env = gym.make(env_id)
    env.set_state_representation(representation)
    env = GymStateWrapper(env)

    np.testing.assert_equal(env.observation_space, env.unwrapped.state_space)

    observation = env.reset()
    np.testing.assert_equal(observation, env.unwrapped.state)
    for _ in range(10):
        action = env.action_space.sample()
        observation, _, done, _ = env.step(action)
        np.testing.assert_equal(observation, env.unwrapped.state)

        if done:
            env.reset()
