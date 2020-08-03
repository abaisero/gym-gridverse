""" Manually control the agent in an environment """

from functools import partial

from gym_gridverse.envs import Actions, Environment
from gym_gridverse.envs.minigrid_env import Minigrid
from gym_gridverse.envs.reset_functions import reset_minigrid_four_rooms
from gym_gridverse.envs.reward_functions import living_reward
from gym_gridverse.envs.reward_functions import reach_goal as reach_goal_reward
from gym_gridverse.envs.state_dynamics import update_agent
from gym_gridverse.envs.terminating_functions import \
    reach_goal as reach_goal_termination
from gym_gridverse.visualize import str_render_state


def get_user_action() -> Actions:
    """Prompts the user for an action input

    Returns:
        Actions: action to take
    """
    while True:
        input_action = input(f"Action? (in (0,{len(Actions)})): ")

        try:
            action = Actions(int(input_action))
        except ValueError:
            pass
        else:
            return action


def manually_control():

    # hard code domain

    domain = Minigrid(
        partial(reset_minigrid_four_rooms, 10, 10),
        update_agent,
        reach_goal_reward,
        reach_goal_termination,
    )

    while True:

        a = get_user_action()
        domain.step(a)
        print(str_render_state(domain.state))


if __name__ == "__main__":
    manually_control()
