from typing import Dict, Iterable, Sequence, Tuple, Type

import numpy as np

from gym_gridverse.debugging import gv_debug
from gym_gridverse.grid_object import Color, GridObject, Hidden, NoneGridObject
from gym_gridverse.observation import Observation
from gym_gridverse.representations.representation import (
    ArrayRepresentation,
    ObservationRepresentation,
    compact_grid_object_representation_convert,
    compact_grid_object_representation_space,
    default_grid_object_representation_convert,
    default_grid_object_representation_space,
    no_overlap_grid_object_representation_convert,
    no_overlap_grid_object_representation_space,
)
from gym_gridverse.representations.spaces import Space
from gym_gridverse.spaces import ObservationSpace


def _sorted_object_types(
    object_types: Iterable[Type[GridObject]],
) -> Sequence[Type[GridObject]]:
    """Returns the object types sorted by their index

    Args:
        object_types (`Iterable[Type[GridObject]]`):

    Returns:
        Sequence[Type[GridObject]]:
    """
    return sorted(object_types, key=lambda obj_type: obj_type.type_index())


def _sorted_colors(colors: Iterable[Color]) -> Sequence[Color]:
    """Returns the colors sorted by their index

    Args:
        colors (`Iterable[Color]`):

    Returns:
        Sequence[Color]:
    """
    return sorted(colors, key=lambda color: color.value)


class ArrayObservationRepresentation(ArrayRepresentation[Observation]):
    def __init__(self, observation_space: ObservationSpace):
        self.observation_space = observation_space


class GridObjectObservationRepresentation(ArrayRepresentation[GridObject]):
    def __init__(self, observation_space: ObservationSpace):
        self.observation_space = observation_space


def make_observation_representation(
    name: str,
    observation_space: ObservationSpace,
) -> ObservationRepresentation:
    """Factory function for observation representations

    Args:
        name (str): name of the representation
        observation_space (ObservationSpace): inner-environment observation space
    Returns:
        ObservationRepresentation:
    """
    # TODO: test

    grid_object_representation: GridObjectObservationRepresentation

    if name == 'default':
        grid_object_representation = DefaultGridObjectObservationRepresentation(
            observation_space
        )
        representations = {
            'grid': GridObservationRepresentation(
                observation_space, grid_object_representation
            ),
            'agent_id_grid': AgentIDGridObservationRepresentation(
                observation_space
            ),
            'item': ItemObservationRepresentation(
                observation_space, grid_object_representation
            ),
        }
        return DictObservationRepresentation(observation_space, representations)

    if name == 'no-overlap':
        # NOTE:  only `grid` and `item` require a separate no-overlap representation
        grid_object_representation = (
            NoOverlapGridObjectObservationRepresentation(observation_space)
        )
        representations = {
            'grid': GridObservationRepresentation(
                observation_space, grid_object_representation
            ),
            'agent_id_grid': AgentIDGridObservationRepresentation(
                observation_space
            ),
            'item': ItemObservationRepresentation(
                observation_space, grid_object_representation
            ),
        }
        return DictObservationRepresentation(observation_space, representations)

    if name == 'compact':
        # NOTE:  only `grid` and `item` require a separate compact representation
        grid_object_representation = CompactGridObjectObservationRepresentation(
            observation_space
        )
        representations = {
            'grid': GridObservationRepresentation(
                observation_space, grid_object_representation
            ),
            'agent_id_grid': AgentIDGridObservationRepresentation(
                observation_space
            ),
            'item': ItemObservationRepresentation(
                observation_space, grid_object_representation
            ),
        }
        return DictObservationRepresentation(observation_space, representations)

    raise ValueError(f'invalid name {name}')


# representation composition


class DictObservationRepresentation(ObservationRepresentation):
    def __init__(
        self,
        observation_space: ObservationSpace,
        representations: Dict[str, ArrayObservationRepresentation],
    ):
        super().__init__(observation_space)
        self.representations = representations

    @property
    def space(self) -> Dict[str, Space]:
        return {
            key: representation.space
            for key, representation in self.representations.items()
        }

    def convert(self, observation: Observation) -> Dict[str, np.ndarray]:
        if gv_debug() and not self.observation_space.contains(observation):
            raise ValueError('observation-space does not contain observation')

        return {
            key: representation.convert(observation)
            for key, representation in self.representations.items()
        }


class GridObservationRepresentation(ArrayObservationRepresentation):
    def __init__(
        self,
        observation_space: ObservationSpace,
        grid_object_representation: GridObjectObservationRepresentation,
    ):
        super().__init__(observation_space)
        self.grid_object_representation = grid_object_representation

    @property
    def space(self) -> Space:
        height = self.observation_space.grid_shape.height
        width = self.observation_space.grid_shape.width

        space_type = self.grid_object_representation.space.space_type
        lower_bound = self.grid_object_representation.space.lower_bound
        lower_bound = np.tile(lower_bound, (height, width, 1))
        upper_bound = self.grid_object_representation.space.upper_bound
        upper_bound = np.tile(upper_bound, (height, width, 1))
        return Space(space_type, lower_bound, upper_bound)

    def convert(self, observation: Observation) -> np.ndarray:
        return np.array(
            [
                [
                    self.grid_object_representation.convert(
                        observation.grid[y, x]
                    )
                    for x in range(observation.grid.shape.width)
                ]
                for y in range(observation.grid.shape.height)
            ],
            int,
        )


class ItemObservationRepresentation(ArrayObservationRepresentation):
    def __init__(
        self,
        observation_space: ObservationSpace,
        grid_object_representation: GridObjectObservationRepresentation,
    ):
        super().__init__(observation_space)
        self.grid_object_representation = grid_object_representation

    @property
    def space(self) -> Space:
        return self.grid_object_representation.space

    def convert(self, observation: Observation) -> np.ndarray:
        return self.grid_object_representation.convert(
            observation.agent.grid_object
        )


class AgentIDGridObservationRepresentation(ArrayObservationRepresentation):
    @property
    def space(self) -> Space:
        height = self.observation_space.grid_shape.height
        width = self.observation_space.grid_shape.width

        if height < 0 or width < 0:
            raise ValueError(f'negative height or width ({height, width})')

        return Space.make_discrete_space(
            np.zeros((height, width), dtype=int),
            np.ones((height, width), dtype=int),
        )

    def convert(self, observation: Observation) -> np.ndarray:
        grid_agent_position = np.zeros(observation.grid.shape.as_tuple, int)
        grid_agent_position[observation.agent.position.yx] = 1
        return grid_agent_position


# grid-object representations


class DefaultGridObjectObservationRepresentation(
    GridObjectObservationRepresentation
):
    """The default representation for a grid-object

    Simply returns the grid-object indices.  See
    :func:`gym_gridverse.representations.representation.default_grid_object_representation_space`
    and
    :func:`gym_gridverse.representations.representation.default_grid_object_convert`
    for more information.
    """

    def __init__(self, observation_space: ObservationSpace):
        super().__init__(observation_space)
        self._grid_object_types = set(self.observation_space.object_types) | {
            Hidden,
            NoneGridObject,
        }
        self._grid_object_colors = set(self.observation_space.colors)

    @property
    def space(self) -> Space:
        return default_grid_object_representation_space(
            self._grid_object_types,
            self._grid_object_colors,
        )

    def convert(self, grid_object: GridObject) -> np.ndarray:
        return default_grid_object_representation_convert(grid_object)


class NoOverlapGridObjectObservationRepresentation(
    GridObjectObservationRepresentation
):
    """The no-overlap representation for a grid-object

    Guarantees that each channel uses separate indices.  See
    :func:`gym_gridverse.representations.representation.no_overlap_grid_object_representation_space`
    and
    :func:`gym_gridverse.representations.representation.no_overlap_grid_object_convert`
    for more information.
    """

    def __init__(self, observation_space: ObservationSpace):
        super().__init__(observation_space)
        self._grid_object_types = set(self.observation_space.object_types) | {
            Hidden,
            NoneGridObject,
        }
        self._grid_object_colors = set(self.observation_space.colors)

    @property
    def space(self) -> Space:
        return no_overlap_grid_object_representation_space(
            self._grid_object_types,
            self._grid_object_colors,
        )

    def convert(self, grid_object: GridObject) -> np.ndarray:
        return no_overlap_grid_object_representation_convert(
            self._grid_object_types,
            self._grid_object_colors,
            grid_object,
        )


class CompactGridObjectObservationRepresentation(
    GridObjectObservationRepresentation
):
    """The compact representation for a grid-object

    Guarantees that each channel uses separate indices, and removes empty gaps
    between indices.  See
    :func:`gym_gridverse.representations.representation.compact_grid_object_representation_space`
    and
    :func:`gym_gridverse.representations.representation.compact_grid_object_convert`
    for more information.
    """

    def __init__(self, observation_space: ObservationSpace):
        super().__init__(observation_space)

        shape: Tuple[int, ...]

        # TODO eventually fix this at the space-level
        max_type_index = self.observation_space.max_type_index
        max_state_index = self.observation_space.max_state_index
        max_color_index = self.observation_space.max_object_color

        shape = (max_type_index + 1,)
        self._grid_object_type_map = -np.ones(shape, int)
        shape = (max_type_index + 1, max_state_index + 1)
        self._grid_object_status_map = -np.ones(shape, int)
        shape = (max_color_index + 1,)
        self._grid_object_color_map = -np.ones(shape, int)

        grid_object_types = _sorted_object_types(
            set(self.observation_space.object_types)
            | {
                Hidden,
                NoneGridObject,
            }
        )
        grid_object_colors = _sorted_colors(self.observation_space.colors)

        compact_index = 0

        for grid_object in grid_object_types:
            i = grid_object.type_index()
            self._grid_object_type_map[i] = compact_index
            compact_index += 1

        for grid_object in grid_object_types:
            i = grid_object.type_index()
            for j in range(grid_object.num_states()):
                self._grid_object_status_map[i, j] = compact_index
                compact_index += 1

        for color in grid_object_colors:
            k = color.value
            self._grid_object_color_map[k] = compact_index
            compact_index += 1

    @property
    def space(self) -> Space:
        return compact_grid_object_representation_space(
            self._grid_object_type_map,
            self._grid_object_status_map,
            self._grid_object_color_map,
        )

    def convert(self, grid_object: GridObject) -> np.ndarray:
        return compact_grid_object_representation_convert(
            self._grid_object_type_map,
            self._grid_object_status_map,
            self._grid_object_color_map,
            grid_object,
        )
