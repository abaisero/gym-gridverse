from typing import Optional, Tuple

from gym_gridverse.envs.inner_env import Actions, InnerEnv
from gym_gridverse.representations.representation import (
    ObservationRepresentation,
    StateRepresentation,
)


class OuterEnv:
    def __init__(
        self,
        env: InnerEnv,
        state_rep: Optional[StateRepresentation] = None,
        observation_rep: Optional[ObservationRepresentation] = None,
    ):
        self.inner_env = env
        self.state_rep = state_rep
        self.observation_rep = observation_rep

        # XXX: rename observation_rep -> observation_repr
        # XXX: rename state_rep -> state_repr

    @property
    def action_space(self):
        return self.inner_env.action_space

    def reset(self):
        self.inner_env.reset()

    def step(self, action: Actions) -> Tuple[float, bool]:
        return self.inner_env.step(action)

    @property
    def state(self):
        if self.state_rep is None:
            raise RuntimeError('State representation not available')

        return self.state_rep.convert(self.inner_env.state)

    @property
    def observation(self):
        if self.observation_rep is None:
            raise RuntimeError('Observation representation not available')

        return self.observation_rep.convert(self.inner_env.observation)
