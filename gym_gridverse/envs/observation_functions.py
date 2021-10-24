import inspect
import itertools as itt
import warnings
from functools import partial
from typing import List, Optional

import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.agent import Agent
from gym_gridverse.envs.visibility_functions import (
    VisibilityFunction,
    visibility_function_registry,
)
from gym_gridverse.geometry import Area, Orientation
from gym_gridverse.grid_object import Hidden
from gym_gridverse.observation import Observation
from gym_gridverse.state import State
from gym_gridverse.utils.functions import (
    checkraise_kwargs,
    import_custom_function,
    is_custom_function,
    select_kwargs,
)
from gym_gridverse.utils.registry import FunctionRegistry


class ObservationFunction(Protocol):
    def __call__(
        self, state: State, *, rng: Optional[rnd.Generator] = None
    ) -> Observation:
        ...


class ObservationFunctionRegistry(FunctionRegistry):
    def get_protocol_parameters(
        self, signature: inspect.Signature
    ) -> List[inspect.Parameter]:
        (state,) = itt.islice(signature.parameters.values(), 1)
        rng = signature.parameters['rng']
        return [state, rng]

    def check_signature(self, function: ObservationFunction):
        signature = inspect.signature(function)
        state, rng = self.get_protocol_parameters(signature)

        # checks first argument is positional ...
        if state.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise ValueError(
                f'The first argument ({state.name}) '
                f'of a registered observation function ({function}) '
                'should be allowed to be a positional argument.'
            )

        # and `rng` is keyword
        if rng.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ]:
            raise ValueError(
                f'The `rng` argument ({rng.name}) '
                f'of a registered observation function ({function}) '
                'should be allowed to be a keyword argument.'
            )

        # checks if annotations, if given, are consistent
        if state.annotation not in [inspect.Parameter.empty, State]:
            warnings.warn(
                f'The first argument ({state.name}) '
                f'of a registered observation function ({function}) '
                f'has an annotation ({state.annotation}) '
                'which is not `State`.'
            )

        if rng.annotation not in [
            inspect.Parameter.empty,
            Optional[rnd.Generator],
        ]:
            warnings.warn(
                f'The `rng` argument ({rng.name}) '
                f'of a registered observation function ({function}) '
                f'has an annotation ({rng.annotation}) '
                'which is not `Optional[rnd.Generator]`.'
            )

        if signature.return_annotation not in [
            inspect.Parameter.empty,
            Observation,
        ]:
            warnings.warn(
                f'The return type of a registered observation function ({function}) '
                f'has an annotation ({signature.return_annotation}) '
                'which is not `Observation`.'
            )


observation_function_registry = ObservationFunctionRegistry()
"""Observation function registry"""


@observation_function_registry.register
def from_visibility(
    state: State,
    *,
    area: Area,
    visibility_function: VisibilityFunction,
    rng: Optional[rnd.Generator] = None,
) -> Observation:
    pov_area = state.agent.get_pov_area(area)
    pov_agent_position = -area.top_left

    observation_grid = state.grid.subgrid(pov_area).change_orientation(
        state.agent.orientation
    )
    visibility = visibility_function(
        observation_grid, pov_agent_position, rng=rng
    )

    if visibility.shape != (area.height, area.width):
        raise ValueError(
            'incorrect visibility shape ({visibility.shape}), should be {(area.height, area.width)}'
        )

    for pos in observation_grid.positions():
        if not visibility[pos.y, pos.x]:
            observation_grid[pos] = Hidden()

    observation_agent = Agent(
        pov_agent_position, Orientation.N, state.agent.obj
    )
    return Observation(observation_grid, observation_agent)


for (name, visibility_function) in visibility_function_registry.items():
    observation_function = partial(
        from_visibility, visibility_function=visibility_function
    )
    observation_function_registry.register(observation_function, name=name)

    # XXX: HACK!!! look away!!!
    globals()[name] = observation_function


def factory(name: str, **kwargs) -> ObservationFunction:

    if is_custom_function(name):
        name = import_custom_function(name)

    try:
        function = observation_function_registry[name]
    except KeyError as error:
        raise ValueError(f'invalid observation function name {name}') from error

    signature = inspect.signature(function)
    required_keys = [
        parameter.name
        for parameter in observation_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is inspect.Parameter.empty
    ]
    optional_keys = [
        parameter.name
        for parameter in observation_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is not inspect.Parameter.empty
    ]

    checkraise_kwargs(kwargs, required_keys)
    kwargs = select_kwargs(kwargs, required_keys + optional_keys)
    return partial(function, **kwargs)
