import functools
import inspect
import warnings
from typing import List, Optional, Union

import numpy as np
import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.geometry import Position
from gym_gridverse.grid import Grid
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.utils.custom import import_if_custom
from gym_gridverse.utils.functions import checkraise_kwargs, select_kwargs
from gym_gridverse.utils.protocols import (
    get_keyword_parameter,
    get_positional_parameters,
)
from gym_gridverse.utils.raytracing import cached_compute_rays_fancy
from gym_gridverse.utils.registry import FunctionRegistry


class VisibilityFunction(Protocol):
    def __call__(
        self,
        grid: Grid,
        position: Position,
        *,
        rng: Optional[rnd.Generator] = None,
    ) -> np.ndarray:
        ...


class VisibilityFunctionRegistry(FunctionRegistry):
    def get_protocol_parameters(
        self, signature: inspect.Signature
    ) -> List[inspect.Parameter]:
        grid, position = get_positional_parameters(signature, 2)
        rng = get_keyword_parameter(signature, 'rng')
        return [grid, position, rng]

    def check_signature(self, function: VisibilityFunction):
        signature = inspect.signature(function)
        grid, position, rng = self.get_protocol_parameters(signature)

        # checks first 2 arguments are positional ...
        if grid.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise TypeError(
                f'The first argument ({grid.name}) '
                f'of a registered visibility function ({function}) '
                'should be allowed to be a positional argument.'
            )

        if position.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ]:
            raise TypeError(
                f'The second argument ({position.name}) '
                f'of a registered visibility function ({function}) '
                'should be allowed to be a positional argument.'
            )

        # and `rng` is keyword
        if rng.kind not in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ]:
            raise TypeError(
                f'The `rng` argument ({rng.name}) '
                f'of a registered visibility function ({function}) '
                'should be allowed to be a keyword argument.'
            )

        # checks if annotations, if given, are consistent
        if grid.annotation not in [inspect.Parameter.empty, Grid]:
            warnings.warn(
                f'The first argument ({grid.name}) '
                f'of a registered visibility function ({function}) '
                f'has an annotation ({grid.annotation}) '
                'which is not `Grid`.'
            )

        if position.annotation not in [inspect.Parameter.empty, Position]:
            warnings.warn(
                f'The second argument ({position.name}) '
                f'of a registered visibility function ({function}) '
                f'has an annotation ({position.annotation}) '
                'which is not `Position`.'
            )

        if rng.annotation not in [
            inspect.Parameter.empty,
            Optional[rnd.Generator],
        ]:
            warnings.warn(
                f'The `rng` argument ({rng.name}) '
                f'of a registered visibility function ({function}) '
                f'has an annotation ({rng.annotation}) '
                'which is not `Optional[rnd.Generator]`.'
            )

        if signature.return_annotation not in [
            inspect.Parameter.empty,
            np.ndarray,
        ]:
            warnings.warn(
                f'The return type of a registered visibility function ({function}) '
                f'has an annotation ({signature.return_annotation}) '
                'which is not `np.ndarray`.'
            )


visibility_function_registry = VisibilityFunctionRegistry()
"""Visibility function registry"""


@visibility_function_registry.register
def fully_transparent(
    grid: Grid, position: Position, *, rng: Optional[rnd.Generator] = None
) -> np.ndarray:
    return np.ones((grid.shape.height, grid.shape.width), dtype=bool)


def _partially_occluded_make_visible(
    visibility, grid, position, next_positions
):
    if grid.area.contains(position) and not visibility[position.y, position.x]:
        visibility[position.y, position.x] = True
        if not grid[position].blocks_vision:
            for next_position in next_positions(position):
                _partially_occluded_make_visible(
                    visibility, grid, next_position, next_positions
                )


def _partially_occluded_next_positions_front_left(position):
    return [
        Position(position.y - 1, position.x),
        Position(position.y, position.x - 1),
        Position(position.y - 1, position.x - 1),
    ]


def _partially_occluded_next_positions_front_right(position):
    return [
        Position(position.y - 1, position.x),
        Position(position.y, position.x + 1),
        Position(position.y - 1, position.x + 1),
    ]


@visibility_function_registry.register
def partially_occluded(
    grid: Grid, position: Position, *, rng: Optional[rnd.Generator] = None
) -> np.ndarray:
    if position.y != grid.shape.height - 1:
        # TODO generalize for this case
        raise NotImplementedError

    visibility_left = np.zeros(
        (grid.shape.height, grid.shape.width), dtype=bool
    )
    _partially_occluded_make_visible(
        visibility_left,
        grid,
        position,
        _partially_occluded_next_positions_front_left,
    )

    visibility_right = np.zeros(
        (grid.shape.height, grid.shape.width), dtype=bool
    )
    _partially_occluded_make_visible(
        visibility_right,
        grid,
        position,
        _partially_occluded_next_positions_front_right,
    )

    visibility = visibility_left | visibility_right
    return visibility


@visibility_function_registry.register
def raytracing(
    grid: Grid,
    position: Position,
    *,
    absolute_counts: bool = True,
    threshold: Union[int, float] = 1,
    rng: Optional[rnd.Generator] = None,
) -> np.ndarray:
    rays = cached_compute_rays_fancy(position, grid.area)
    counts_num = np.zeros((grid.shape.height, grid.shape.width), dtype=int)
    counts_den = np.zeros((grid.shape.height, grid.shape.width), dtype=int)

    for ray in rays:
        light = True
        for pos in ray:
            counts_num[pos.y, pos.x] += int(light)
            counts_den[pos.y, pos.x] += 1
            light = light and not grid[pos].blocks_vision

    visibility = (
        counts_num >= threshold
        if absolute_counts
        else (counts_num / counts_den) >= threshold
    )
    return visibility


@visibility_function_registry.register
def stochastic_raytracing(  # TODO: add test
    grid: Grid,
    position: Position,
    *,
    rng: Optional[rnd.Generator] = None,
) -> np.ndarray:
    rng = get_gv_rng_if_none(rng)

    rays = cached_compute_rays_fancy(position, grid.area)
    counts_num = np.zeros((grid.shape.height, grid.shape.width), dtype=int)
    counts_den = np.zeros((grid.shape.height, grid.shape.width), dtype=int)

    for ray in rays:
        light = True
        for pos in ray:
            counts_num[pos.y, pos.x] += int(light)
            counts_den[pos.y, pos.x] += 1
            light = light and not grid[pos].blocks_vision

    probs = np.nan_to_num(counts_num / counts_den)
    visibility = rng.random(probs.shape) <= probs
    return visibility


def factory(name: str, **kwargs) -> VisibilityFunction:
    name = import_if_custom(name)

    try:
        function = visibility_function_registry[name]
    except KeyError as error:
        raise ValueError(f'invalid visibility function name {name}') from error

    signature = inspect.signature(function)
    required_keys = [
        parameter.name
        for parameter in visibility_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is inspect.Parameter.empty
    ]
    optional_keys = [
        parameter.name
        for parameter in visibility_function_registry.get_nonprotocol_parameters(
            signature
        )
        if parameter.default is not inspect.Parameter.empty
    ]

    checkraise_kwargs(kwargs, required_keys)
    kwargs = select_kwargs(kwargs, required_keys + optional_keys)
    return functools.partial(function, **kwargs)
