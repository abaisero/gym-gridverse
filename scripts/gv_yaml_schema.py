import argparse
import json

from gym_gridverse.envs.yaml.schemas import schemas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indent', type=int, default=4)
    args = parser.parse_args()

    json_schema = schemas['env'].json_schema('TO-BE-REMOVED')
    # remove mandatory $id field ('TO-BE-REMOVED')
    del json_schema['$id']

    print(json.dumps(json_schema, indent=args.indent))


if __name__ == '__main__':
    main()
