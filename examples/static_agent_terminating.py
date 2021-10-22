from gym_gridverse.action import Action
from gym_gridverse.envs.terminating_functions import (
    terminating_function_registry,
)
from gym_gridverse.state import State


@terminating_function_registry.register
def static_agent(state: State, action: Action, next_state: State) -> bool:
    return state.agent == next_state.agent
