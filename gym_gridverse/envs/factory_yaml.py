from functools import partial
from os import path
from typing import Dict, List, Optional, TextIO, Type

import yamale

from gym_gridverse import grid_object
from gym_gridverse.actions import Actions
from gym_gridverse.envs import observation_functions as observation_fs
from gym_gridverse.envs import reset_functions as reset_fs
from gym_gridverse.envs import reward_functions as reward_fs
from gym_gridverse.envs import state_dynamics as transition_fs
from gym_gridverse.envs import terminating_functions as terminating_fs
from gym_gridverse.envs.env import Environment
from gym_gridverse.envs.gridworld import GridWorld
from gym_gridverse.geometry import DistanceFunction, Position, Shape
from gym_gridverse.spaces import (
    ActionSpace,
    DomainSpace,
    ObservationSpace,
    StateSpace,
)


def make_schema() -> yamale.schema.Schema:
    schema_path = path.join(path.dirname(__file__), 'assets/schema.yaml')
    return yamale.make_schema(schema_path)


def make_data(stream: TextIO, *, validate=True) -> Dict:
    schema = make_schema()
    content = stream.read()
    data = yamale.make_data(content=content)

    if validate:
        validate_data(data, schema)

    return data[0][0]  # weird yamale format


def make_environment(stream: TextIO, *, validate=True) -> Environment:
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


def make_environment_from_data(data: Dict) -> Environment:
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
    objects = list(map(make_object_type, data['objects']))
    colors = list(map(make_color, data['colors']))
    return StateSpace(shape, objects, colors)


def make_action_space(data: Optional[List[str]]):
    if data is None:
        actions = list(Actions)
    else:
        if len(set(data)) != len(data):
            raise ValueError('Duplicate actions')

        actions = list(map(make_action, data))

    return ActionSpace(actions)


def make_observation_space(data: Dict):
    shape = Shape(*data['shape'])
    objects = list(map(make_object_type, data['objects']))
    colors = list(map(make_color, data['colors']))
    return ObservationSpace(shape, objects, colors)


def make_reset_function(
    data: Dict, *, state_space: StateSpace
) -> reset_fs.ResetFunction:

    size = (
        state_space.grid_shape.height
        if state_space.grid_shape.height == state_space.grid_shape.width
        else None
    )

    return reset_fs.factory(
        data['name'],
        height=state_space.grid_shape.height,
        width=state_space.grid_shape.width,
        size=size,
        random_agent_pos=data.get('random_agent'),
        num_obstacles=data.get('num_obstacles'),
    )


def make_transition_function(data: Dict) -> transition_fs.StateDynamics:
    transition_functions = list(map(_make_transition_function, data))

    def transition_function(s, a):
        for f in transition_functions:
            f(s, a)

    return transition_function


def _make_transition_function(data: Dict) -> transition_fs.StateDynamics:
    return transition_fs.factory(data['name'])


def make_reward_function(data: Dict) -> reward_fs.RewardFunction:
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
        object_type = make_object_type(data_object_type)

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
        visibility_function = make_visibility_function(data_visibility_function)

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
        object_type = make_object_type(data_object_type)

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


def make_visibility_function(name) -> observation_fs.VisibilityFunction:
    if name == 'minigrid':
        return observation_fs.minigrid_visibility

    if name == 'raytracing':
        return observation_fs.raytracing_visibility

    if name == 'stochastic_raytracing':
        return observation_fs.stochastic_raytracing_visibility

    raise ValueError(f'invalid visibility function name `{name}`')


def make_object_type(name) -> Type[grid_object.GridObject]:
    try:
        return getattr(grid_object, name)
    except AttributeError:
        raise ValueError(f'invalid object type `{name}`')


def make_action(name) -> Actions:
    try:
        return getattr(Actions, name)
    except AttributeError:
        raise ValueError(f'invalid action name `{name}`')


def make_color(name) -> grid_object.Colors:
    try:
        return getattr(grid_object.Colors, name)
    except AttributeError:
        raise ValueError(f'invalid color name `{name}`')
