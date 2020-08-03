import copy
from typing import Tuple

from gym_gridverse.envs import Actions, Environment
from gym_gridverse.envs.reset_functions import ResetFunction
from gym_gridverse.envs.reward_functions import RewardFunction
from gym_gridverse.envs.state_dynamics import StateDynamics
from gym_gridverse.envs.terminating_functions import TerminatingFunction
from gym_gridverse.spaces import ActionSpace, ObservationSpace, StateSpace
from gym_gridverse.state import State


class Minigrid(Environment):
    def __init__(
        self,
        reset_function: ResetFunction,
        step_function: StateDynamics,
        reward_function: RewardFunction,
        termination_function: TerminatingFunction,
    ):

        self._functional_reset = reset_function
        self._functional_step = step_function
        self.reward_function = reward_function
        self.termination_function = termination_function


        # TODO: fix spaces
        # TODO: fix python
        super().__init__(StateSpace(), ActionSpace(), ObservationSpace())

    def functional_reset(self) -> State:
        return self._functional_reset()

    def functional_step(
        self, state: State, action: Actions
    ) -> Tuple[State, float, bool]:

        next_state = copy.deepcopy(state)
        self._functional_step(next_state, action)

        reward = self.reward_function(state, action, next_state)
        terminal = self.termination_function(state, action, next_state)

        return (next_state, reward, terminal)
