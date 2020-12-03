from functools import lru_cache

from schema import And, Optional, Or, Schema

from gym_gridverse.actions import Actions
from gym_gridverse.grid_object import Colors

# general purpose schemas


@lru_cache()
def pair_schema():
    return Schema(lambda data: len(data) == 2, error='{} should have length 2')


@lru_cache()
def non_empty_schema():
    return Schema(len, error='{} should not be empty')


@lru_cache()
def at_least_2_schema():
    return Schema(
        lambda data: len(data) >= 2, error='{} should have at least 2 elements'
    )


@lru_cache()
def positive_schema():
    return Schema(lambda data: data > 0, error='{} should be positive')


@lru_cache()
def odd_second_schema():
    return Schema(
        lambda data: data[1] % 2 == 1,
        error='{} should have an odd second element',
    )


@lru_cache()
def odd_width_schema():
    return Schema(
        lambda data: data[1] % 2 == 1,
        error='{} should have an odd width element',
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
            pair_schema(),
        ),
        description='A pair of positive integers',
    )


@lru_cache()
def layout_schema():
    return Schema(
        And(
            [And(int, positive_schema())],
            pair_schema(),
        ),
        description='A pair of positive integers',
    )


@lru_cache()
def object_type_schema():
    return Schema(
        Or(
            'floor',
            'wall',
            'goal',
            'door',
            'key',
            'moving_obstacle',
            'box',
            # XXX remove these
            'Floor',
            'Wall',
            'Goal',
            'Door',
            'Key',
            'MovingObstacle',
            'Box',
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
            [color.name for color in Colors],
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
            [action.name for action in Actions],
            at_least_2_schema(),
            unique_schema(),
        ),
        description='A non-empty list of unique action names',
    )


@lru_cache()
def observation_space_schema():
    return Schema(
        {
            'shape': And(shape_schema(), odd_width_schema()),
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
        },
        name='reset_function',
        as_reference=True,
    )


@lru_cache()
def transition_function_schema():
    return Schema(
        {'name': str},
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
