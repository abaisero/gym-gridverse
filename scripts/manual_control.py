""" Manually control the agent in an environment """

import argparse

from gym_gridverse.envs import Actions, Environment
from gym_gridverse.envs.factory import (
    STRING_TO_GYM_CONSTRUCTOR,
    gym_minigrid_from_descr,
)
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


def manually_control(domain: Environment):
    domain.reset()
    while True:

        a = get_user_action()

        r, t = domain.step(a)

        if t:
            print("Resetting environment")
            domain.reset()

        print(f"Reward {r}, next state:\n{str_render_state(domain.state)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'descr',
        help=f"Gym description, available: {list(STRING_TO_GYM_CONSTRUCTOR.keys())}",
    )
    manually_control(gym_minigrid_from_descr(parser.parse_args().descr))
