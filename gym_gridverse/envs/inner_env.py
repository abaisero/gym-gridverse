import abc
from typing import Optional, Tuple

from gym_gridverse.action import Action
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import ActionSpace, ObservationSpace, StateSpace
from gym_gridverse.state import State

__all__ = ['InnerEnv']


class InnerEnv(metaclass=abc.ABCMeta):
    """Inner environment

    Inner environments provide an interface primarily based on python objects,
    with states represented by :py:class:`~gym_gridverse.state.State`,
    observations by :py:class:`~gym_gridverse.observation.Observation`, and
    actions by :py:class:`~gym_gridverse.action.Action`.

    """

    def __init__(
        self,
        state_space: StateSpace,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.observation_space = observation_space

        self._state: Optional[State] = None
        self._observation: Optional[Observation] = None

    @abc.abstractmethod
    def set_seed(self, seed: Optional[int] = None):
        assert False, "Must be implemented by derived class"

    @abc.abstractmethod
    def functional_reset(self) -> State:
        """Returns a new state"""
        assert False, "Must be implemented by derived class"

    @abc.abstractmethod
    def functional_step(
        self, state: State, action: Action
    ) -> Tuple[State, float, bool]:
        """Returns next state, reward, and done flag"""
        assert False, "Must be implemented by derived class"

    @abc.abstractmethod
    def functional_observation(self, state: State) -> Observation:
        """Returns observation"""
        assert False, "Must be implemented by derived class"

    def reset(self):
        """Resets the state

        Internally calls :py:meth:`functional_reset` to reset the state;  also
        resets the observation, so that an updated observation will be
        generated upon request.
        """
        self._state = self.functional_reset()
        self._observation = None

    def step(self, action: Action) -> Tuple[float, bool]:
        """Runs the dynamics for one timestep, and returns reward and done flag

        Internally calls :py:meth:`functional_step` to update the state;  also
        resets the observation, so that an updated observation will be
        generated upon request.

        Args:
            action (Action): the chosen action to apply

        Returns:
            Tuple[float, bool]: reward and terminal
        """

        self._state, reward, done = self.functional_step(self.state, action)
        self._observation = None
        return reward, done

    @property
    def state(self) -> State:
        """Return the current state

        Returns:
            State:
        """
        if self._state is None:
            raise RuntimeError(
                'The state was not set properly;  was the environment reset?'
            )

        return self._state

    @property
    def observation(self) -> Observation:
        """Returns the current observation

        Internally calls :py:meth:`functional_observation` to generate the
        current observation based on the current state.  The observation is
        generated lazily, such that at most one observation is generated for
        each state.  As a consequence, this will return the same observation
        until the state is reset/updated, even if the observation function is
        stochastic.

        Returns:
            Observation:
        """
        # memoizing observation because observation function can be stochastic
        if self._observation is None:
            self._observation = self.functional_observation(self.state)

        return self._observation
