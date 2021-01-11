import gym

class GymStateObservationWrapper(gym.Wrapper):
    """
    Gym Wrapper to replace the standard observation representation with state instead.

    Doesn't change underlying environment, won't change render
    """
    def __init__(self, env):
        super().__init__(env)
        assert env.state_space is not None  # Make sure we have a valid state representation
        self.observation_space = env.state_space


    def step(self, action: int):
        _, r, d, info = self.env.step(action)
        return self.env.state, r, d, info

    def reset(self):
        self.env.reset()
        return self.env.state