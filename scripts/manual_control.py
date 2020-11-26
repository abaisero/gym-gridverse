""" Manually control the agent in an environment """

import argparse

from gym_gridverse.actions import Actions
from gym_gridverse.envs import InnerEnv
from gym_gridverse.envs.factory import (STRING_TO_GYM_CONSTRUCTOR,
                                        gym_minigrid_from_descr)
from gym_gridverse.visualize import str_render_obs, str_render_state


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


def manually_control(domain: InnerEnv):
    domain.reset()
    while True:

        a = get_user_action()

        r, t = domain.step(a)

        if t:
            print("Resetting environment")
            domain.reset()

        state = domain.state
        obs = domain.observation

        print(
            f"Reward {r}, "
            f"next state:\n{str_render_state(state)}\n"
            f"observation:\n{str_render_obs(obs)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'descr',
        help=f"Gym description, available: {list(STRING_TO_GYM_CONSTRUCTOR.keys())}",
    )
    manually_control(gym_minigrid_from_descr(parser.parse_args().descr))
