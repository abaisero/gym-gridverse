import importlib.resources
from functools import partial
from typing import Dict, List, Optional, Sequence, TextIO

import numpy.random as rnd
import yamale

from gym_gridverse import grid_object
from gym_gridverse.actions import Actions
from gym_gridverse.envs import (
    observation_functions as observation_fs,
    reset_functions as reset_fs,
    reward_functions as reward_fs,
    terminating_functions as terminating_fs,
    transition_functions as transition_fs,
    visibility_functions as visibility_fs,
)
from gym_gridverse.envs.gridworld import GridWorld
from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.geometry import DistanceFunction, Position, Shape
from gym_gridverse.grid_object import Colors
from gym_gridverse.spaces import (
    ActionSpace,
    DomainSpace,
    ObservationSpace,
    StateSpace,
)
from gym_gridverse.state import State


def make_schema() -> yamale.schema.Schema:
    content = importlib.resources.read_text(
        'gym_gridverse.envs.resources', 'schema.yaml'
    )
    return yamale.make_schema(content=content)


def make_data(stream: TextIO, *, validate=True) -> Dict:
    schema = make_schema()
    content = stream.read()
    data = yamale.make_data(content=content)

    if validate:
        validate_data(data, schema)

    return data[0][0]  # weird yamale format


def make_environment(stream: TextIO, *, validate=True) -> InnerEnv:
    data = make_data(stream, validate=validate)
    return make_environment_from_data(data)


def validate_data(data: Dict, schema: yamale.schema.Schema):
    try:
        yamale.validate(schema, data)

    except yamale.YamaleError as e:
        print('Validation failed!')
        for result in e.results:
            print(
                f'Error validating data `{result.data}` with `{result.schema}`'
            )
            print('Error list:')
            for i, error in enumerate(result.errors):
                print(f'{i}) {error}')

        raise


def make_environment_from_data(data: Dict) -> InnerEnv:
    state_space = make_state_space(data['state_space'])
    action_space = make_action_space(data.get('action_space'))
    observation_space = make_observation_space(data['observation_space'])

    reset_function = make_reset_function(data['reset'], state_space=state_space)
    transition_function = make_transition_function(data['transition'])
    reward_function = make_reward_function(data['reward'])
    observation_function = make_observation_function(
        data['observation'], observation_space=observation_space
    )
    terminating_function = make_terminating_function(data['terminating'])

    return GridWorld(
        DomainSpace(state_space, action_space, observation_space),
        reset_function,
        transition_function,
        observation_function,
        reward_function,
        terminating_function,
    )


def make_state_space(data: Dict):
    shape = Shape(*data['shape'])
    objects = list(map(grid_object.factory_type, data['objects']))
    colors = [Colors[name] for name in data['colors']]
    return StateSpace(shape, objects, colors)


def make_action_space(data: Optional[List[str]]):
    if data is None:
        actions = list(Actions)
    else:
        if len(set(data)) != len(data):
            raise ValueError('Duplicate actions')

        actions = list(map(Actions.__getitem__, data))

    return ActionSpace(actions)


def make_observation_space(data: Dict):
    shape = Shape(*data['shape'])
    objects = list(map(grid_object.factory_type, data['objects']))
    colors = list(map(Colors.__getitem__, data['colors']))
    return ObservationSpace(shape, objects, colors)


def make_reset_function(
    data: Dict, *, state_space: StateSpace
) -> reset_fs.ResetFunction:

    size = (
        state_space.grid_shape.height
        if state_space.grid_shape.height == state_space.grid_shape.width
        else None
    )

    try:
        data_object_type = data['object_type']
    except KeyError:
        object_type = None
    else:
        object_type = grid_object.factory_type(data_object_type)

    return reset_fs.factory(
        data['name'],
        height=state_space.grid_shape.height,
        width=state_space.grid_shape.width,
        size=size,
        layout=data.get('layout'),
        random_agent_pos=data.get('random_agent'),
        num_obstacles=data.get('num_obstacles'),
        num_rivers=data.get('num_rivers'),
        object_type=object_type,
    )


def make_transition_function(
    data: Sequence[Dict],
) -> transition_fs.TransitionFunction:
    transition_functions = list(map(_make_transition_function, data))

    def transition_function(
        state: State, action: Actions, *, rng: Optional[rnd.Generator] = None
    ) -> None:
        for f in transition_functions:
            f(state, action, rng=rng)

    return transition_function


def _make_transition_function(data: Dict) -> transition_fs.TransitionFunction:
    return transition_fs.factory(data['name'])


def make_reward_function(data: Sequence[Dict]) -> reward_fs.RewardFunction:
    reward_functions = list(map(_make_reward_function, data))
    return partial(reward_fs.chain, reward_functions=reward_functions)


def _make_reward_function(data: Dict) -> reward_fs.RewardFunction:

    try:
        data_reward_functions = data['reward_functions']
    except KeyError:
        reward_functions = None
    else:
        reward_functions = list(
            map(_make_reward_function, data_reward_functions)
        )

    try:
        data_object_type = data['object_type']
    except KeyError:
        object_type = None
    else:
        object_type = grid_object.factory_type(data_object_type)

    try:
        data_distance_function = data['distance_function']
    except KeyError:
        distance_function = None
    else:
        distance_function = make_distance_function(data_distance_function)

    return reward_fs.factory(
        data['name'],
        reward_functions=reward_functions,
        reward=data.get('reward'),
        reward_on=data.get('reward_on'),
        reward_off=data.get('reward_off'),
        object_type=object_type,
        distance_function=distance_function,
        reward_per_unit_distance=data.get('reward_per_unit_distance'),
        reward_closer=data.get('reward_closer'),
        reward_further=data.get('reward_further'),
    )


def make_observation_function(
    data: Dict, *, observation_space: ObservationSpace
) -> observation_fs.ObservationFunction:

    try:
        data_visibility_function = data['visibility_function']
    except KeyError:
        visibility_function = None
    else:
        visibility_function = visibility_fs.factory(data_visibility_function)

    return observation_fs.factory(
        data['name'],
        observation_space=observation_space,
        visibility_function=visibility_function,
    )


def make_terminating_function(data: Dict) -> terminating_fs.TerminatingFunction:

    try:
        data_terminating_functions = data['terminating_functions']
    except KeyError:
        terminating_functions = None
    else:
        terminating_functions = list(
            map(make_terminating_function, data_terminating_functions)
        )

    try:
        data_object_type = data['object_type']
    except KeyError:
        object_type = None
    else:
        object_type = grid_object.factory_type(data_object_type)

    return terminating_fs.factory(
        data['name'],
        terminating_functions=terminating_functions,
        object_type=object_type,
    )


def make_distance_function(name) -> DistanceFunction:
    if name == 'manhattan':
        return Position.manhattan_distance

    if name == 'euclidean':
        return Position.euclidean_distance

    raise ValueError(f'invalid distance function name `{name}`')
