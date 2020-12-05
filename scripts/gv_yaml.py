#!/usr/bin/env python
import argparse

from gym_gridverse.envs.yaml.factory import factory_env_from_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='YAML data file')
    args = parser.parse_args()

    env = factory_env_from_yaml(args.path)
    print(env)


if __name__ == '__main__':
    main()
