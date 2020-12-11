"""Print random interaction with environment to terminal"""

import argparse
import random
from typing import Dict

import numpy as np

from gym_gridverse.actions import Actions
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.render_as_string import str_render_obs, str_render_state
from gym_gridverse.representations.observation_representations import (
    DefaultObservationRepresentation,
)
from gym_gridverse.spaces import ActionSpace


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

        # typically not used outside of library, inner representation used to
        # visualize below
        internal_state = domain.inner_env.state
        internal_obs = domain.inner_env.observation

        print(
            f"Reward {r}, "
            f"next state:\n{str_render_state(internal_state)}\n"
            f"observation:\n{str_render_obs(internal_obs)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env_path', help='env YAML file')
    args = parser.parse_args()

    inner_env = factory_env_from_yaml(args.env_path)

    rep = DefaultObservationRepresentation(inner_env.observation_space)
    outer_env = OuterEnv(inner_env, observation_rep=rep)

    visualize_random_bot(outer_env)
