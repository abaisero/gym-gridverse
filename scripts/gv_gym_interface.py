#!/usr/bin/env python
import argparse

import gym

import gym_gridverse
from gym_gridverse.visualize import str_render_obs, str_render_state


def main(args):
    gym_env = gym.make(args.env_id)
    assert isinstance(gym_env, gym_gridverse.gym.GymEnvironment)

    while True:
        done = False

        gym_env.reset()
        while not done:
            # TODO perform visualization via the gym_env.render() API
            str_state = str_render_state(gym_env.outer_env.inner_env.state)
            str_observation = str_render_obs(
                gym_env.outer_env.inner_env.observation
            )

            print('next state:')
            print(f'{str_state}')
            print('observation:')
            print(f'{str_observation}')

            action = gym_env.action_space.sample()
            str_action = gym_env.outer_env.action_space.int_to_action(
                action
            ).name
            input(
                f'Agent will take action {str_action}, press a key to continue'
            )

            _, reward, done, _ = gym_env.step(action)
            print(f'Reward {reward}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'env_id', choices=gym_gridverse.gym.env_ids, help='Gym environment id',
    )
    main(parser.parse_args())
