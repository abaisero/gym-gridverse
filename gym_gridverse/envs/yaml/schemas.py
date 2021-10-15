from functools import lru_cache

from schema import And, Optional, Or, Schema

from gym_gridverse.action import Action
from gym_gridverse.grid_object import Color

# general purpose schemas


@lru_cache()
def non_empty_schema():
    return Schema(len, error='{} should not be empty')


@lru_cache()
def len_schema(n: int):
    return Schema(
        lambda data: len(data) == n, error=f'{{}} should have length {n}'
    )


@lru_cache()
def len_at_least_n_schema(n: int):
    return Schema(
        lambda data: len(data) >= n,
        error=f'{{}} should have at least {n} elements',
    )


@lru_cache()
def positive_schema():
    return Schema(lambda data: data > 0, error='{} should be positive')


@lru_cache()
def odd_element_schema(i: int):
    return Schema(
        lambda data: data[i] % 2 == 1,
        error=f'element {i} of {{}} should be odd',
    )


@lru_cache()
def unique_schema():
    return Schema(
        lambda data: len(set(data)) == len(data),
        error='{} should have unique elements',
    )


# gridverse concept schemas


@lru_cache()
def shape_schema():
    return Schema(
        And(
            [And(int, positive_schema())],
            len_schema(2),
        ),
        description='A pair of positive integers',
    )


@lru_cache()
def layout_schema():
    return Schema(
        And(
            [And(int, positive_schema())],
            len_schema(2),
        ),
        description='A pair of positive integers',
    )


@lru_cache()
def object_type_schema():
    return Schema(
        Or(
            'floor',
            'wall',
            'exit',
            'door',
            'key',
            'moving_obstacle',
            'box',
            'telepod',
            'beacon',
            # XXX remove these
            'Floor',
            'Wall',
            'Exit',
            'Door',
            'Key',
            'MovingObstacle',
            'Box',
            'Telepod',
            'Beacon',
        )
    )


@lru_cache()
def object_types_schema():
    return Schema(
        And(
            [object_type_schema()],
            non_empty_schema(),
            unique_schema(),
        ),
        description='A non-empty list of unique grid-object names',
    )


@lru_cache()
def colors_schema():
    return Schema(
        And(
            [color.name for color in Color],
            non_empty_schema(),
            unique_schema(),
        ),
        description='A non-empty list of unique color names',
    )


# space schemas


@lru_cache()
def state_space_schema():
    return Schema(
        {
            'shape': shape_schema(),
            'objects': object_types_schema(),
            'colors': colors_schema(),
        },
        description='The shape and contents of a state',
    )


@lru_cache()
def action_space_schema():
    return Schema(
        And(
            [action.name for action in Action],
            len_at_least_n_schema(2),
            unique_schema(),
        ),
        description='A non-empty list of unique action names',
    )


@lru_cache()
def observation_space_schema():
    return Schema(
        {
            'shape': And(shape_schema(), odd_element_schema(1)),
            'objects': object_types_schema(),
            'colors': colors_schema(),
        },
        description='The shape and contents of an observation;  shape should have an off width.',
    )


# function schemas


@lru_cache()
def reset_function_schema():
    return Schema(
        {
            'name': str,
            Optional('random_agent'): bool,
            Optional('layout'): layout_schema(),
            Optional('num_obstacles'): And(int, positive_schema()),
            Optional('random_agent'): bool,
            Optional('num_rivers'): And(int, positive_schema()),
            Optional('object_type'): object_type_schema(),
            Optional('colors'): colors_schema(),
            Optional('num_beacons'): And(int, positive_schema()),
            Optional('num_exits'): And(int, positive_schema()),
            Optional('kwargs'): dict,
        },
        name='reset_function',
        as_reference=True,
    )


@lru_cache()
def transition_function_schema():
    return Schema(
        {
            'name': str,
            Optional('kwargs'): dict,
        },
        description='A transition function',
        name='transition_function',
        as_reference=True,
    )


@lru_cache()
def transition_functions_schema():
    return Schema(
        And(
            [transition_function_schema()],
            non_empty_schema(),
        ),
        description='A list of transition functions',
    )


@lru_cache()
def distance_function_schema():
    return Schema(
        Or('manhattan', 'euclidean'),
        description='A distance function',
    )


@lru_cache()
def _reward_function_schemas():
    rf_schema = Schema(
        {
            'name': str,
            Optional('reward'): float,
            Optional('reward_on'): float,
            Optional('reward_off'): float,
            Optional('reward_per_unit_distance'): float,
            Optional('reward_closer'): float,
            Optional('reward_further'): float,
            Optional('distance_function'): distance_function_schema(),
            Optional('object_type'): object_type_schema(),
            Optional('reward_open'): float,
            Optional('reward_close'): float,
            Optional('reward_pick'): float,
            Optional('reward_drop'): float,
            Optional('reward_good'): float,
            Optional('reward_bad'): float,
            Optional('kwargs'): dict,
        },
        description='A reward function',
        name='reward_function',
        as_reference=True,
    )

    rfs_schema = Schema(
        And(
            [rf_schema],
            non_empty_schema(),
        ),
        description='A list of reward functions',
    )

    # injecting recursive definition
    rf_schema.schema[Optional('reward_functions')] = rfs_schema

    return rf_schema, rfs_schema


@lru_cache()
def reward_function_schema():
    schema, _ = _reward_function_schemas()
    return schema


@lru_cache()
def reward_functions_schema():
    _, schema = _reward_function_schemas()
    return schema


@lru_cache()
def visibility_function_schema():
    return Schema(
        Or(
            'full_visibility',
            'minigrid_visibility',
            'raytracing_visibility',
            'stochastic_raytracing_visibility',
        ),
        description='A visibility functions',
    )


@lru_cache()
def observation_function_schema():
    return Schema(
        {
            'name': str,
            Optional('visibility_function'): visibility_function_schema(),
        },
        description='An observation function',
        name='observation_function',
        as_reference=True,
    )


@lru_cache()
def terminating_function_schema():
    schema = Schema(
        {
            'name': str,
            Optional('object_type'): object_type_schema(),
            Optional('kwargs'): dict,
        },
        description='A terminating function',
        name='terminating_function',
        as_reference=True,
    )

    # injecting recursive definition
    schema.schema[Optional('terminating_functions')] = Schema(
        [schema],
        description='A list of terminating functions',
    )
    return schema


@lru_cache()
def env_schema():
    return Schema(
        {
            'state_space': state_space_schema(),
            Optional('action_space'): action_space_schema(),
            'observation_space': observation_space_schema(),
            'reset_function': reset_function_schema(),
            'transition_functions': transition_functions_schema(),
            'reward_functions': reward_functions_schema(),
            'observation_function': observation_function_schema(),
            'terminating_function': terminating_function_schema(),
        },
    )
