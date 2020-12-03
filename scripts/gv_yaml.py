#!/usr/bin/env python
import argparse

from gym_gridverse.envs.factory_yaml import GridVerseValidator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='YAML data file')
    args = parser.parse_args()

    env = GridVerseValidator().make_env_from_yaml(args.data_path)
    print(env)


if __name__ == '__main__':
    main()
