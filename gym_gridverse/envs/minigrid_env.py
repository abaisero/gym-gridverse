from typing import Callable, Dict, Tuple

from gym_gridverse.env import Actions, Environment
from gym_gridverse.env.state_dynamics import step_objects, update_agent
from gym_gridverse.spaces import ActionSpace, ObservationSpace, StateSpace
from gym_gridverse.state import State


class Minigrid(Environment):
    def __init__(self, reset_function: Callable[[], State]):
        # TODO: fix spaces
        super().__init__(StateSpace(), ActionSpace(), ObservationSpace())

        self._functional_reset = reset_function

    def functional_reset(self) -> State:
        return self._functional_reset()

    def functional_step(
        self, state: State, action: Actions
    ) -> Tuple[State, float, bool, Dict]:

        for state_dynamics in [update_agent, step_objects]:
            state_dynamics(state, action)

        raise NotImplementedError
