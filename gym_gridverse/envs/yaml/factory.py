from typing import List, Tuple, Type

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
from gym_gridverse.envs.yaml.schemas import schemas
from gym_gridverse.geometry import (
    DistanceFunction,
    Shape,
    distance_function_factory,
)
from gym_gridverse.grid_object import Color, GridObject, factory_type
from gym_gridverse.space_builders import StateSpaceBuilder
from gym_gridverse.spaces import (
    ActionSpace,
    DomainSpace,
    ObservationSpace,
    StateSpace,
)


def process_reserved_keys(data):

    if 'transition_functions' in data:
        data['transition_functions'] = [
            factory_transition_function(d) for d in data['transition_functions']
        ]

    if 'reward_functions' in data:
        data['reward_functions'] = [
            factory_reward_function(d) for d in data['reward_functions']
        ]

    if 'terminating_functions' in data:
        data['terminating_functions'] = [
            factory_terminating_function(d)
            for d in data['terminating_functions']
        ]

    if 'reward_function' in data:
        data['reward_function'] = factory_reward_function(
            data['reward_function']
        )

    if 'distance_function' in data:
        data['distance_function'] = factory_distance_function(
            data['distance_function']
        )

    if 'visibility_function' in data:
        data['visibility_function'] = factory_visibility_function(
            data['visibility_function']
        )

    if 'shape' in data:
        data['shape'] = Shape(*data['shape'])

    if 'layout' in data:
        data['layout'] = tuple(data['layout'])

    if 'object_type' in data:
        data['object_type'] = factory_type(data['object_type'])

    if 'colors' in data:
        data['colors'] = set(factory_colors(data['colors']))


def factory_shape(data) -> Shape:
    data = schemas['shape'].validate(data)
    return Shape(*data)


def factory_layout(data) -> Tuple[int, int]:
    data = schemas['layout'].validate(data)
    return tuple(data)  # type: ignore


def factory_object_types(data) -> List[Type[GridObject]]:
    data = schemas['object_types'].validate(data)
    return [factory_type(d) for d in data]


def factory_colors(data) -> List[Color]:
    data = schemas['colors'].validate(data)
    return [Color[name] for name in data]


def factory_distance_function(data) -> DistanceFunction:
    data = schemas['distance_function'].validate(data)
    return distance_function_factory(data)


def factory_state_space_builder(data) -> StateSpaceBuilder:
    data = schemas['state_space'].validate(data)
    objects = factory_object_types(data['objects'])
    colors = factory_colors(data['colors'])

    state_space_builder = StateSpaceBuilder()
    state_space_builder.set_object_types(objects)
    state_space_builder.set_colors(colors)
    return state_space_builder


def factory_action_space(data) -> ActionSpace:
    data = schemas['action_space'].validate(data)

    return ActionSpace([Action[name] for name in data])


def factory_observation_space(data) -> ObservationSpace:
    data = schemas['observation_space'].validate(data)

    shape = factory_shape(data['shape'])
    objects = factory_object_types(data['objects'])
    colors = factory_colors(data['colors'])

    return ObservationSpace(
        grid_shape=shape,
        object_types=objects,
        colors=colors,
    )


def factory_reset_function(data) -> reset_fs.ResetFunction:
    data = schemas['reset_function'].validate(data)

    name = data.pop('name')
    process_reserved_keys(data)
    return reset_fs.factory(
        name,
        **data,
    )


def factory_transition_function(data) -> transition_fs.TransitionFunction:
    data = schemas['transition_function'].validate(data)

    name = data.pop('name')
    process_reserved_keys(data)
    return transition_fs.factory(name, **data)


def factory_reward_function(data) -> reward_fs.RewardFunction:
    data = schemas['reward_function'].validate(data)

    name = data.pop('name')
    process_reserved_keys(data)
    return reward_fs.factory(name, **data)


def factory_visibility_function(data) -> visibility_fs.VisibilityFunction:
    # TODO: test, maybe? (re-check coverage)
    data = schemas['visibility_function'].validate(data)

    name = data.pop('name')
    process_reserved_keys(data)
    return visibility_fs.factory(name, **data)


def factory_observation_function(
    data, observation_space: ObservationSpace
) -> observation_fs.ObservationFunction:
    # TODO: test, maybe? (re-check coverage)
    data = schemas['observation_function'].validate(data)

    name = data.pop('name')
    process_reserved_keys(data)
    return observation_fs.factory(
        name,
        observation_space=observation_space,
        **data,
    )


def factory_terminating_function(data) -> terminating_fs.TerminatingFunction:
    # TODO: test, maybe? (re-check coverage)
    data = schemas['terminating_function'].validate(data)

    name = data.pop('name')
    process_reserved_keys(data)
    return terminating_fs.factory(name, **data)


def factory_env_from_data(data) -> InnerEnv:
    data = schemas['env'].validate(data)

    state_space_builder = factory_state_space_builder(data['state_space'])
    action_space = (
        factory_action_space(data['action_space'])
        if 'action_space' in data
        else ActionSpace(list(Action))
    )
    observation_space = factory_observation_space(data['observation_space'])

    reset_function = factory_reset_function(data['reset_function'])
    transition_function = factory_transition_function(
        {'name': 'chain', 'transition_functions': data['transition_functions']}
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

    state_space_builder.set_grid_shape(reset_function().grid.shape)
    state_space = state_space_builder.build()

    domain_space = DomainSpace(state_space, action_space, observation_space)

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
