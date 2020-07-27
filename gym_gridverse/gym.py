import gym
from gym_gridverse.env import Environment


class GridVerseEnvironment(gym.Env):  # pylint: disable=abstract-method
    def __init__(self, env: Environment):
        super().__init__()
        self.env = env

        self.action_space = as_gym_space(env.action_space)
        self.observation_space = as_gym_space(env.observation_space)

    def reset(self):
        self.env.reset()
        observation = self.env.get_observation()
        return self._convert_observation(observation)

    def step(self, action: int):
        reward, done, info = self.env.step(action)
        observation = self.env.get_observation()
        return self._convert_observation(observation), reward, done, info

    def _convert_observation(self, observation):
        # TODO do conversion
        gym_observation = observation

        return gym_observation


def as_gym_space(space):
    # TODO do conversion
    gym_space = space

    # TODO as gym.space.Dict or whatever

    return gym_space
