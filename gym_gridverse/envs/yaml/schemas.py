import itertools as itt
from functools import lru_cache

from schema import And, Optional, Or, Schema

from gym_gridverse.action import Action
from gym_gridverse.grid_object import Color

# general purpose schemas


@lru_cache()
def _non_empty_schema():
    return Schema(len, error='{} should not be empty')


@lru_cache()
def _len_schema(n: int):
    return Schema(
        lambda data: len(data) == n, error=f'{{}} should have length {n}'
    )


@lru_cache()
def _positive_schema():
    return Schema(lambda data: data > 0, error='{} should be positive')


@lru_cache()
def _odd_element_schema(i: int):
    return Schema(
        lambda data: data[i] % 2 == 1,
        error=f'element {i} of {{}} should be odd',
    )


@lru_cache()
def _pair(subschema):
    return Schema(And([subschema], _len_schema(2)))


@lru_cache()
def _positive_int_pair():
    return _pair(Schema(And(int, _positive_schema())))


@lru_cache()
def _unique_schema():
    return Schema(
        lambda data: len(set(data)) == len(data),
        error='{} should have unique elements',
    )


# TODO this should be evaluated at runtime so that it is up-to-date with
# registered GridObjects

# base schemas
schemas = {
    'shape': _positive_int_pair(),
    'layout': _positive_int_pair(),
    'area': _pair(_pair(int)),
    'object_type': Schema(str),
    'action': Schema(Or(*(action.name for action in Action))),
    'color': Schema(Or(*(color.name for color in Color))),
}

# more base schemas
schemas.update(
    {
        'object_types': Schema(
            And(
                [schemas['object_type']],
                _non_empty_schema(),
                _unique_schema(),
            ),
            description='A non-empty list of unique grid-object type names',
        ),
        'actions': Schema(
            And(
                [schemas['action']],
                _non_empty_schema(),
                _unique_schema(),
            ),
            description='A non-empty list of unique actions',
        ),
        'colors': Schema(
            And(
                [schemas['color']],
                _non_empty_schema(),
                _unique_schema(),
            ),
            description='A non-empty list of unique color names',
        ),
    }
)

# add function schemas
schemas.update(
    {
        'reset_function': Schema(
            {'name': str, Optional(object): object},
            description='A reset function',
            name='reset_function',
            as_reference=True,
        ),
        'transition_function': Schema(
            {'name': str, Optional(object): object},
            description='A transition function',
            name='transition_function',
            as_reference=True,
        ),
        'reward_function': Schema(
            {'name': str, Optional(object): object},
            description='A reward function',
            name='reward_function',
            as_reference=True,
        ),
        'observation_function': Schema(
            {'name': str, Optional(object): object},
            description='An observation function',
            name='observation_function',
            as_reference=True,
        ),
        'visibility_function': Schema(
            {'name': str, Optional(object): object},
            description='A visibility function',
            name='visibility_function',
            as_reference=True,
        ),
        'terminating_function': Schema(
            {'name': str, Optional(object): object},
            description='A terminating function',
            name='terminating_function',
            as_reference=True,
        ),
        #
        'distance_function': Schema(
            Or('manhattan', 'euclidean'),
            description='A distance function',
        ),
    }
)

# add function list schemas
schemas.update(
    {
        'reset_functions': Schema(
            And(
                [schemas['reset_function']],
                _non_empty_schema(),
            ),
            description='A list of reset functions',
        ),
        'transition_functions': Schema(
            And(
                [schemas['transition_function']],
                _non_empty_schema(),
            ),
            description='A list of transition functions',
        ),
        'reward_functions': Schema(
            And(
                [schemas['reward_function']],
                _non_empty_schema(),
            ),
            description='A list of reward functions',
        ),
        'terminating_functions': Schema(
            And(
                [schemas['terminating_function']],
                _non_empty_schema(),
            ),
            description='A list of terminating functions',
        ),
    }
)

schema_keys = [
    'reset_function',
    'transition_function',
    'reward_function',
    'observation_function',
    'visibility_function',
    'terminating_function',
]
reserved_keys = [
    'reset_function',
    'transition_function',
    'reward_function',
    'terminating_function',
    #
    'reset_functions',
    'transition_functions',
    'reward_functions',
    'terminating_functions',
    #
    'shape',
    'layout',
    'object_type',
    'colors',
]
for schema_key, reserved_key in itt.product(schema_keys, reserved_keys):
    schema = schemas[schema_key]
    schema.schema[Optional(reserved_key)] = schemas[reserved_key]


# space schemas
schemas.update(
    {
        'state_space': Schema(
            {
                'objects': schemas['object_types'],
                'colors': schemas['colors'],
            },
            description='The shape and contents of a state',
        ),
        'action_space': Schema(
            schemas['actions'],
            description='A non-empty list of unique action names',
        ),
        'observation_space': Schema(
            {
                'objects': schemas['object_types'],
                'colors': schemas['colors'],
            },
            description='The shape and contents of an observation;  shape should have an odd width.',
        ),
    }
)

# env schema
schemas.update(
    {
        'env': Schema(
            {
                'state_space': schemas['state_space'],
                Optional('action_space'): schemas['action_space'],
                'observation_space': schemas['observation_space'],
                'reset_function': schemas['reset_function'],
                'transition_functions': schemas['transition_functions'],
                'reward_functions': schemas['reward_functions'],
                'observation_function': schemas['observation_function'],
                'terminating_function': schemas['terminating_function'],
            },
        )
    }
)
