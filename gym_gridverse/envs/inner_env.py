import abc
from typing import Optional, Tuple

from gym_gridverse.action import Action
from gym_gridverse.debugging import checkraise
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import ActionSpace, ObservationSpace, StateSpace
from gym_gridverse.state import State

__all__ = ['InnerEnv']


class InnerEnv(metaclass=abc.ABCMeta):
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
        assert False, "Must be implemented by derived class"

    @abc.abstractmethod
    def functional_step(
        self, state: State, action: Action
    ) -> Tuple[State, float, bool]:
        assert False, "Must be implemented by derived class"

    @abc.abstractmethod
    def functional_observation(self, state: State) -> Observation:
        assert False, "Must be implemented by derived class"

    def reset(self):
        self._state = self.functional_reset()
        self._observation = None

    def step(self, action: Action) -> Tuple[float, bool]:
        """Updates the state by applying `action`

        Calls :py:meth:`functional_step` under the hood on :py:meth:`state` and
        resets :py:meth:`observation` to ensure the next observation will be
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
        """Returns the current state

        The state is the result of a sequence of
        :py:class:`~gym_gridverse.action.Action` through the :py:meth:`step`
        function.

        To reset the state, see :py:meth:`reset`

        Returns:
            State: the current state of the environment
        """
        checkraise(
            lambda: self._state is not None,
            RuntimeError,
            'The state was not set properly;  was the environment reset?',
        )

        return self._state

    @property
    def observation(self) -> Observation:
        """Returns the current observation

        This implicitly calls :py:meth:`functional_observation`, which
        generates an observation from the current :py:meth:`state`.

        NOTE: even when using stochastic observation functions this call will
        always return the same values.

        Returns:
            Observation:
        """
        # memoizing observation because observation function can be stochastic
        if self._observation is None:
            self._observation = self.functional_observation(self.state)

        return self._observation
