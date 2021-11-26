from typing import Dict, Optional, Tuple

import numpy as np

from gym_gridverse.envs.inner_env import Action, InnerEnv
from gym_gridverse.representations.representation import (
    ObservationRepresentation,
    StateRepresentation,
)
from gym_gridverse.spaces import ActionSpace


class OuterEnv:
    """Outer environment

    Outer environments provide an interface primarily based on numeric data,
    with states and observations represented by :py:class:`~numpy.ndarray`, and
    actions by :py:class:`~gym_gridverse.action.Action`.

    """

    def __init__(
        self,
        env: InnerEnv,
        *,
        state_representation: Optional[StateRepresentation] = None,
        observation_representation: Optional[ObservationRepresentation] = None,
    ):
        self.inner_env = env
        self.state_representation = state_representation
        self.observation_representation = observation_representation

    @property
    def action_space(self) -> ActionSpace:
        """Returns the action space of the problem.

        Returns:
            ActionSpace:
        """
        return self.inner_env.action_space

    def reset(self) -> None:
        """Resets the state"""
        self.inner_env.reset()

    def step(self, action: Action) -> Tuple[float, bool]:
        """Runs the dynamics for one timestep, and returns reward and done flag

        Args:
            action(~gym_gridverse.action.Action): agent's action

        Returns:
            Tuple[float, bool]: (reward, terminality)
        """
        return self.inner_env.step(action)

    @property
    def state(self) -> Dict[str, np.ndarray]:
        """Returns the representation of the current state.

        Returns:
            Dict[str, numpy.ndarray]:
        """
        if self.state_representation is None:
            raise RuntimeError('State representation not available')

        return self.state_representation.convert(self.inner_env.state)

    @property
    def observation(self) -> Dict[str, np.ndarray]:
        """Returns the representation of the current observation.

        Returns:
            Dict[str, numpy.ndarray]:
        """
        if self.observation_representation is None:
            raise RuntimeError('Observation representation not available')

        return self.observation_representation.convert(
            self.inner_env.observation
        )
