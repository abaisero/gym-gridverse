#!/usr/bin/env python
import argparse

from gym_gridverse.envs.factory_yaml import make_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='YAML data file')
    args = parser.parse_args()

    with open(args.data_path) as f:
        env = make_environment(f)

    print(env)


if __name__ == '__main__':
    main()
