""" Manually control the agent in an environment """

import argparse
import random
from typing import Dict

import numpy as np
from gym_gridverse.actions import Actions
from gym_gridverse.envs.factory import (STRING_TO_GYM_CONSTRUCTOR,
                                        gym_minigrid_from_descr)
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import \
    DefaultObservationRepresentation
from gym_gridverse.spaces import ActionSpace
from gym_gridverse.visualize import str_render_obs, str_render_state


def random_action_selection(
    _observation: Dict[str, np.ndarray], action_space: ActionSpace
) -> Actions:
    """Simply returns a random action"""
    return random.choice(action_space.actions)


def visualize_random_bot(domain: OuterEnv):
    domain.reset()
    while True:

        a = random_action_selection(domain.observation, domain.action_space)
        input(f"Agent will take action {a.name}, press a key to continue")

        r, t = domain.step(a)

        if t:
            print("Resetting environment")
            domain.reset()

        internal_state = domain.env.state
        internal_obs = domain.env.observation

        print(
            f"Reward {r}, "
            f"next state:\n{str_render_state(internal_state)}\n"
            f"observation:\n{str_render_obs(internal_obs)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'descr',
        help=f"Gym description: {list(STRING_TO_GYM_CONSTRUCTOR.keys())}",
    )

    inner_env = gym_minigrid_from_descr(parser.parse_args().descr)
    rep = DefaultObservationRepresentation(inner_env.observation_space)
    outer_env = OuterEnv(inner_env, obs_rep=rep)

    visualize_random_bot(outer_env)
