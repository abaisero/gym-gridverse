from typing import Optional, Tuple

from gym_gridverse.envs.env import Actions, Environment
from gym_gridverse.representations.representation import Representation


class OuterEnv:
    def __init__(
        self,
        env: Environment,
        state_rep: Optional[Representation],
        obs_rep: Optional[Representation],
    ):
        self.env = env
        self.state_rep = state_rep
        self.obs_rep = obs_rep

    def reset(self):
        self.env.reset()

    def step(self, action: Actions) -> Tuple[float, bool]:
        return self.env.step(action)

    @property
    def observation(self):
        return self.obs_rep(self.env.observation)

    @property
    def state(self):
        return self.state_rep(self.env.state)
