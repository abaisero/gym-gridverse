#!/usr/bin/env python
import argparse

import gym

import gym_gridverse


def main(args):
    gym_env = gym.make(args.env_id)
    assert isinstance(gym_env, gym_gridverse.gym.GymEnvironment)

    while True:
        done = False

        gym_env.reset()
        gym_env.render()

        while not done:
            action_int = gym_env.action_space.sample()
            action = gym_env.outer_env.action_space.int_to_action(action_int)

            input(
                f'Agent will take action {action.name}, press a key to continue'
            )

            _, reward, done, _ = gym_env.step(action_int)
            gym_env.render()

            print(f'Reward {reward}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'env_id',
        choices=gym_gridverse.gym.env_ids,
        help='Gym environment id',
    )
    main(parser.parse_args())
