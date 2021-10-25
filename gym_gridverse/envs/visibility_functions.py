import functools
import inspect
import warnings
from typing import List, Optional, Union

import numpy as np
import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.geometry import (
    Area,
    Position,
    StrideDirection,
    diagonal_strides,
)
from gym_gridverse.grid import Grid
from gym_gridverse.rng import get_gv_rng_if_none
from gym_gridverse.utils.functions import (
    checkraise_kwargs,
    import_custom_function,
    is_custom_function,
    select_kwargs,
)
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


@visibility_function_registry.register
def partially_occluded(
    grid: Grid, position: Position, *, rng: Optional[rnd.Generator] = None
) -> np.ndarray:

    if position.y != grid.shape.height - 1:
        #  gym-minigrid does not handle this case, and we are not currently
        #  generalizing it
        raise NotImplementedError

    visibility = np.zeros((grid.shape.height, grid.shape.width), dtype=bool)
    visibility[position.y, position.x] = True  # agent

    # front
    x = position.x
    for y in range(position.y - 1, -1, -1):
        visibility[y, x] = visibility[y + 1, x] and grid[y + 1, x].transparent

    # right
    y = position.y
    for x in range(position.x + 1, grid.shape.width):
        visibility[y, x] = visibility[y, x - 1] and grid[y, x - 1].transparent

    # left
    y = position.y
    for x in range(position.x - 1, -1, -1):
        visibility[y, x] = visibility[y, x + 1] and grid[y, x + 1].transparent

    # top left
    positions = diagonal_strides(
        Area(
            (0, position.y - 1),
            (0, position.x - 1),
        ),
        StrideDirection.NW,
    )
    for p in positions:
        visibility[p.y, p.x] = (
            (grid[p.y + 1, p.x].transparent and visibility[p.y + 1, p.x])
            or (grid[p.y, p.x + 1].transparent and visibility[p.y, p.x + 1])
            or (
                grid[p.y + 1, p.x + 1].transparent
                and visibility[p.y + 1, p.x + 1]
            )
        )

    # top right
    positions = diagonal_strides(
        Area(
            (0, position.y - 1),
            (position.x + 1, grid.shape.width - 1),
        ),
        StrideDirection.NE,
    )
    for p in positions:
        visibility[p.y, p.x] = (
            (grid[p.y + 1, p.x].transparent and visibility[p.y + 1, p.x])
            or (grid[p.y, p.x - 1].transparent and visibility[p.y, p.x - 1])
            or (
                grid[p.y + 1, p.x - 1].transparent
                and visibility[p.y + 1, p.x - 1]
            )
        )

    return visibility


@visibility_function_registry.register
def minigrid(
    grid: Grid, position: Position, *, rng: Optional[rnd.Generator] = None
) -> np.ndarray:

    if position.y != grid.shape.height - 1:
        #  gym-minigrid does not handle this case, and we are not currently
        #  generalizing it
        raise NotImplementedError

    visibility = np.zeros((grid.shape.height, grid.shape.width), dtype=bool)
    visibility[position.y, position.x] = True  # agent

    for y in range(grid.shape.height - 1, -1, -1):
        for x in range(grid.shape.width - 1):
            if visibility[y, x] and grid[y, x].transparent:
                visibility[y, x + 1] = True
                if y > 0:
                    visibility[y - 1, x] = True
                    visibility[y - 1, x + 1] = True

        for x in range(grid.shape.width - 1, 0, -1):
            if visibility[y, x] and grid[y, x].transparent:
                visibility[y, x - 1] = True
                if y > 0:
                    visibility[y - 1, x] = True
                    visibility[y - 1, x - 1] = True

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
            light = light and grid[pos].transparent

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
            light = light and grid[pos].transparent

    probs = np.nan_to_num(counts_num / counts_den)
    visibility = rng.random(probs.shape) <= probs
    return visibility


def factory(name: str, **kwargs) -> VisibilityFunction:

    if is_custom_function(name):
        name = import_custom_function(name)

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
