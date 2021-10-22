from typing import Dict, Optional, Tuple

import numpy as np

from gym_gridverse.debugging import checkraise
from gym_gridverse.envs.inner_env import Action, InnerEnv
from gym_gridverse.representations.representation import (
    ObservationRepresentation,
    StateRepresentation,
)
from gym_gridverse.spaces import ActionSpace


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
    def action_space(self) -> ActionSpace:
        """Returns the action space of the problem

        Returns:
            ActionSpace:
        """
        return self.inner_env.action_space

    def reset(self) -> None:
        """Resets the state of the environment"""
        self.inner_env.reset()

    def step(self, action: Action) -> Tuple[float, bool]:
        """Updates the state according to the action

        Args:
            action: agent's action

        Returns:
            Tuple[float, bool]: (reward, terminality)
        """
        return self.inner_env.step(action)

    @property
    def state(self) -> Dict[str, np.ndarray]:
        """Returns the representation of the current state

        Returns:
            Dict[str, numpy.ndarray]:
        """
        checkraise(
            lambda: self.state_rep is not None,
            RuntimeError,
            'State representation not available',
        )

        return self.state_rep.convert(self.inner_env.state)

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        """Returns the representation of the current observation

        Returns:
            Dict[str, numpy.ndarray]:
        """
        checkraise(
            lambda: self.observation_rep is not None,
            RuntimeError,
            'Observation representation not available',
        )

        return self.observation_rep.convert(self.inner_env.observation)
