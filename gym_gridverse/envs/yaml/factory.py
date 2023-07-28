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
    Area,
    DistanceFunction,
    Shape,
    distance_function_factory,
)
from gym_gridverse.grid_object import Color, GridObject, grid_object_registry
from gym_gridverse.spaces import ActionSpace
from gym_gridverse.utils.custom import import_if_custom
from gym_gridverse.utils.space_builders import (
    ObservationSpaceBuilder,
    StateSpaceBuilder,
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

    if 'area' in data:
        data['area'] = Area(*data['area'])

    if 'object_type' in data:
        data['object_type'] = grid_object_registry.from_name(
            data['object_type']
        )

    if 'colors' in data:
        data['colors'] = set(factory_colors(data['colors']))


def factory_shape(data) -> Shape:
    data = schemas['shape'].validate(data)
    return Shape(*data)


def factory_layout(data) -> Tuple[int, int]:
    data = schemas['layout'].validate(data)
    layout_y, layout_x = data
    return (layout_y, layout_x)


def factory_object_type(data) -> Type[GridObject]:
    data = schemas['object_type'].validate(data)
    name = import_if_custom(data)
    return grid_object_registry.from_name(name)


def factory_object_types(data) -> List[Type[GridObject]]:
    data = schemas['object_types'].validate(data)
    return [factory_object_type(d) for d in data]


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


def factory_observation_space_builder(data) -> ObservationSpaceBuilder:
    data = schemas['observation_space'].validate(data)
    objects = factory_object_types(data['objects'])
    colors = factory_colors(data['colors'])

    observation_space_builder = ObservationSpaceBuilder()
    observation_space_builder.set_object_types(objects)
    observation_space_builder.set_colors(colors)
    return observation_space_builder


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


def factory_observation_function(data) -> observation_fs.ObservationFunction:
    # TODO: test, maybe? (re-check coverage)
    data = schemas['observation_function'].validate(data)

    name = data.pop('name')
    process_reserved_keys(data)
    return observation_fs.factory(name, **data)


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
    observation_space_builder = factory_observation_space_builder(
        data['observation_space']
    )

    reset_function = factory_reset_function(data['reset_function'])
    transition_function = factory_transition_function(
        {'name': 'chain', 'transition_functions': data['transition_functions']}
    )
    reward_function = factory_reward_function(
        {'name': 'reduce_sum', 'reward_functions': data['reward_functions']}
    )
    observation_function = factory_observation_function(
        data['observation_function']
    )
    terminating_function = factory_terminating_function(
        data['terminating_function']
    )

    state = reset_function()
    state_space_builder.set_grid_shape(state.grid.shape)
    state_space = state_space_builder.build()

    observation = observation_function(state)
    observation_space_builder.set_grid_shape(observation.grid.shape)
    observation_space = observation_space_builder.build()

    return GridWorld(
        state_space,
        action_space,
        observation_space,
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
