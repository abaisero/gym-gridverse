from typing import Optional, Tuple

from gym_gridverse.envs.env import Actions, Environment
from gym_gridverse.representations.representation import (
    ObservationRepresentation,
    StateRepresentation,
)


class OuterEnv:
    def __init__(
        self,
        env: Environment,
        state_rep: Optional[StateRepresentation] = None,
        obs_rep: Optional[ObservationRepresentation] = None,
    ):
        self.env = env
        self.state_rep = state_rep
        self.obs_rep = obs_rep

        # XXX: rename obs_rep -> observation_repr
        # XXX: rename state_rep -> state_repr
        # XXX: rename Env -> Environment

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        self.env.reset()

    def step(self, action: Actions) -> Tuple[float, bool]:
        return self.env.step(action)

    @property
    def state(self):
        if self.state_rep is None:
            raise RuntimeError('State representation not available')

        return self.state_rep.convert(self.env.state)

    @property
    def observation(self):
        if self.obs_rep is None:
            raise RuntimeError('Observation representation not available')

        return self.obs_rep.convert(self.env.observation)
