""" Manually control the agent in an environment """

from gym_gridverse.env import Actions, Environment
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

    # Initialize environment
    domain: Environment = 0

    while True:

        a = get_user_action()
        domain.step(a)
        print(str_render_state(domain.state))


if __name__ == "__main__":
    manually_control()
