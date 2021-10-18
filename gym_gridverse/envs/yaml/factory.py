from functools import partial
from typing import List, Optional, Tuple, Type

import yaml
from gym_gridverse.action import Action
from gym_gridverse.envs import (
    InnerEnv,
    observation_functions as observation_fs,
    reset_functions as reset_fs,
    reward_functions as reward_fs,
    terminating_functions as terminating_fs,
    transition_functions as transition_fs,
    visibility_functions as visibility_fs,
)
from gym_gridverse.envs.gridworld import GridWorld
from gym_gridverse.envs.yaml import schemas
from gym_gridverse.geometry import (
    DistanceFunction,
    Shape,
    distance_function_factory,
)
from gym_gridverse.grid_object import Color, GridObject, factory_type
from gym_gridverse.spaces import (
    ActionSpace,
    DomainSpace,
    ObservationSpace,
    StateSpace,
)


def factory_shape(data) -> Shape:
    data = schemas.shape_schema().validate(data)
    return Shape(*data)


def factory_layout(data) -> Tuple[int, int]:
    data = schemas.layout_schema().validate(data)
    return tuple(data)  # type: ignore


def factory_object_types(data) -> List[Type[GridObject]]:
    data = schemas.object_types_schema().validate(data)
    return [factory_type(d) for d in data]


def factory_colors(data) -> List[Color]:
    data = schemas.colors_schema().validate(data)
    return [Color[name] for name in data]


def factory_distance_function(data) -> DistanceFunction:
    data = schemas.distance_function_schema().validate(data)
    return distance_function_factory(data)


def factory_state_space(data) -> StateSpace:
    data = schemas.state_space_schema().validate(data)

    shape = factory_shape(data['shape'])
    objects = factory_object_types(data['objects'])
    colors = factory_colors(data['colors'])

    return StateSpace(
        grid_shape=shape,
        object_types=objects,
        colors=colors,
    )


def factory_action_space(data) -> ActionSpace:
    data = schemas.action_space_schema().validate(data)

    return ActionSpace([Action[name] for name in data])


def factory_observation_space(data) -> ObservationSpace:
    data = schemas.observation_space_schema().validate(data)

    shape = factory_shape(data['shape'])
    objects = [factory_type(name) for name in data['objects']]
    colors = [Color[name] for name in data['colors']]

    return ObservationSpace(
        grid_shape=shape,
        object_types=objects,
        colors=colors,
    )


def factory_reset_function(
    data, state_space: StateSpace
) -> reset_fs.ResetFunction:
    data = schemas.reset_function_schema().validate(data)

    layout: Optional[Tuple[int, int]]
    try:
        data_layout = data['layout']
    except KeyError:
        layout = None
    else:
        layout = tuple(data_layout)  # type: ignore

    try:
        data_object_type = data['object_type']
    except KeyError:
        object_type = None
    else:
        object_type = factory_type(data_object_type)

    try:
        data_colors = data['colors']
    except KeyError:
        colors = None
    else:
        colors = set(factory_colors(data_colors))

    return reset_fs.factory(
        data['name'],
        height=state_space.grid_shape.height,
        width=state_space.grid_shape.width,
        layout=layout,
        random_agent_pos=data.get('random_agent'),
        num_obstacles=data.get('num_obstacles'),
        num_rivers=data.get('num_rivers'),
        object_type=object_type,
        colors=colors,
        num_beacons=data.get('num_beacons'),
        num_exits=data.get('num_exits'),
    )


def factory_transition_function(data) -> transition_fs.TransitionFunction:
    data = schemas.transition_functions_schema().validate(data)

    transition_functions = [transition_fs.factory(d['name']) for d in data]
    return partial(
        transition_fs.chain, transition_functions=transition_functions
    )


def factory_reward_function(data) -> reward_fs.RewardFunction:
    data = schemas.reward_function_schema().validate(data)

    try:
        data_reward_functions = data['reward_functions']
    except KeyError:
        reward_functions = None
    else:
        reward_functions = [
            factory_reward_function(d) for d in data_reward_functions
        ]

    try:
        data_object_type = data['object_type']
    except KeyError:
        object_type = None
    else:
        object_type = factory_type(data_object_type)

    try:
        data_distance_function = data['distance_function']
    except KeyError:
        distance_function = None
    else:
        distance_function = factory_distance_function(data_distance_function)

    return reward_fs.factory(
        data['name'],
        reward_functions=reward_functions,
        object_type=object_type,
        distance_function=distance_function,
        reward=data.get('reward'),
        reward_on=data.get('reward_on'),
        reward_off=data.get('reward_off'),
        reward_per_unit_distance=data.get('reward_per_unit_distance'),
        reward_closer=data.get('reward_closer'),
        reward_further=data.get('reward_further'),
        reward_open=data.get('reward_open'),
        reward_close=data.get('reward_close'),
        reward_pick=data.get('reward_pick'),
        reward_drop=data.get('reward_drop'),
        reward_good=data.get('reward_good'),
        reward_bad=data.get('reward_bad'),
    )


def factory_visibility_function(data):
    # TODO: test
    data = schemas.visibility_function_schema().validate(data)
    return visibility_fs.factory(data['name'])


def factory_observation_function(
    data, observation_space: ObservationSpace
) -> observation_fs.ObservationFunction:
    data = schemas.observation_function_schema().validate(data)

    try:
        data_visibility_function = data['visibility_function']
    except KeyError:
        visibility_function = None
    else:
        # TODO: test
        visibility_function = factory_visibility_function(
            data_visibility_function
        )

    return observation_fs.factory(
        data['name'],
        observation_space=observation_space,
        visibility_function=visibility_function,
    )


def factory_terminating_function(data) -> terminating_fs.TerminatingFunction:
    data = schemas.terminating_function_schema().validate(data)

    try:
        data_terminating_functions = data['terminating_functions']
    except KeyError:
        terminating_functions = None
    else:
        terminating_functions = [
            factory_terminating_function(d) for d in data_terminating_functions
        ]

    try:
        data_object_type = data['object_type']
    except KeyError:
        object_type = None
    else:
        # TODO: test
        object_type = factory_type(data_object_type)

    return terminating_fs.factory(
        data['name'],
        terminating_functions=terminating_functions,
        object_type=object_type,
    )


def factory_env_from_data(data) -> InnerEnv:
    data = schemas.env_schema().validate(data)

    state_space = factory_state_space(data['state_space'])

    try:
        data_action_space = data['action_space']
    except KeyError:
        action_space = ActionSpace(list(Action))
    else:
        action_space = factory_action_space(data_action_space)

    observation_space = factory_observation_space(data['observation_space'])

    domain_space = DomainSpace(state_space, action_space, observation_space)

    reset_function = factory_reset_function(data['reset_function'], state_space)
    transition_function = factory_transition_function(
        data['transition_functions']
    )
    reward_function = factory_reward_function(
        {'name': 'reduce_sum', 'reward_functions': data['reward_functions']}
    )
    observation_function = factory_observation_function(
        data['observation_function'], observation_space
    )
    terminating_function = factory_terminating_function(
        data['terminating_function']
    )

    return GridWorld(
        domain_space,
        reset_function,
        transition_function,
        observation_function,
        reward_function,
        terminating_function,
    )


def factory_env_from_yaml(path: str) -> InnerEnv:
    with open(path) as f:
        data = yaml.safe_load(f)

    return factory_env_from_data(data)
