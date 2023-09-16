from typing import Dict, Iterable, Sequence, Tuple, Type

import numpy as np

from gym_gridverse.debugging import gv_debug
from gym_gridverse.grid_object import Color, GridObject, NoneGridObject
from gym_gridverse.representations.representation import (
    ArrayRepresentation,
    StateRepresentation,
    compact_grid_object_representation_convert,
    compact_grid_object_representation_space,
    default_grid_object_representation_convert,
    default_grid_object_representation_space,
    no_overlap_grid_object_representation_convert,
    no_overlap_grid_object_representation_space,
)
from gym_gridverse.representations.spaces import Space
from gym_gridverse.spaces import StateSpace
from gym_gridverse.state import State


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


class ArrayStateRepresentation(ArrayRepresentation[State]):
    def __init__(self, state_space: StateSpace):
        self.state_space = state_space


class GridObjectStateRepresentation(ArrayRepresentation[GridObject]):
    def __init__(self, state_space: StateSpace):
        self.state_space = state_space


def make_state_representation(
    name: str,
    state_space: StateSpace,
) -> StateRepresentation:
    """Factory function for state representations

    Args:
        name (str): name of the representation
        state_space (StateSpace): inner-environment state space
    Returns:
        StateRepresentation:
    """
    # TODO: test

    grid_object_representation: GridObjectStateRepresentation

    if name == 'default':
        grid_object_representation = DefaultGridObjectStateRepresentation(
            state_space
        )
        representations = {
            'grid': GridStateRepresentation(
                state_space, grid_object_representation
            ),
            'agent_id_grid': AgentIDGridStateRepresentation(state_space),
            'agent': AgentStateRepresentation(state_space),
            'item': ItemStateRepresentation(
                state_space, grid_object_representation
            ),
        }
        return DictStateRepresentation(state_space, representations)

    if name == 'no-overlap':
        # NOTE:  only `grid` and `item` require a separate no-overlap representation
        grid_object_representation = NoOverlapGridObjectStateRepresentation(
            state_space
        )
        representations = {
            'grid': GridStateRepresentation(
                state_space, grid_object_representation
            ),
            'agent_id_grid': AgentIDGridStateRepresentation(state_space),
            'agent': AgentStateRepresentation(state_space),
            'item': ItemStateRepresentation(
                state_space, grid_object_representation
            ),
        }
        return DictStateRepresentation(state_space, representations)

    if name == 'compact':
        # NOTE:  only `grid` and `item` require a separate compact representation
        grid_object_representation = CompactGridObjectStateRepresentation(
            state_space
        )
        representations = {
            'grid': GridStateRepresentation(
                state_space, grid_object_representation
            ),
            'agent_id_grid': AgentIDGridStateRepresentation(state_space),
            'agent': AgentStateRepresentation(state_space),
            'item': ItemStateRepresentation(
                state_space, grid_object_representation
            ),
        }
        return DictStateRepresentation(state_space, representations)

    raise ValueError(f'invalid name {name}')


# representation composition


class DictStateRepresentation(StateRepresentation):
    def __init__(
        self,
        state_space: StateSpace,
        representations: Dict[str, ArrayStateRepresentation],
    ):
        super().__init__(state_space)
        self.representations = representations

    @property
    def space(self) -> Dict[str, Space]:
        return {
            key: representation.space
            for key, representation in self.representations.items()
        }

    def convert(self, state: State) -> Dict[str, np.ndarray]:
        if gv_debug() and not self.state_space.contains(state):
            raise ValueError('state-space does not contain state')

        return {
            key: representation.convert(state)
            for key, representation in self.representations.items()
        }


# dict field representations


class GridStateRepresentation(ArrayStateRepresentation):
    def __init__(
        self,
        state_space: StateSpace,
        grid_object_representation: GridObjectStateRepresentation,
    ):
        super().__init__(state_space)
        self.grid_object_representation = grid_object_representation

    @property
    def space(self) -> Space:
        height = self.state_space.grid_shape.height
        width = self.state_space.grid_shape.width

        space_type = self.grid_object_representation.space.space_type
        lower_bound = self.grid_object_representation.space.lower_bound
        lower_bound = np.tile(lower_bound, (height, width, 1))
        upper_bound = self.grid_object_representation.space.upper_bound
        upper_bound = np.tile(upper_bound, (height, width, 1))
        return Space(space_type, lower_bound, upper_bound)

    def convert(self, state: State) -> np.ndarray:
        return np.array(
            [
                [
                    self.grid_object_representation.convert(state.grid[y, x])
                    for x in range(state.grid.shape.width)
                ]
                for y in range(state.grid.shape.height)
            ],
            int,
        )


class ItemStateRepresentation(ArrayStateRepresentation):
    def __init__(
        self,
        state_space: StateSpace,
        grid_object_representation: GridObjectStateRepresentation,
    ):
        super().__init__(state_space)
        self.grid_object_representation = grid_object_representation

    @property
    def space(self) -> Space:
        return self.grid_object_representation.space

    def convert(self, state: State) -> np.ndarray:
        return self.grid_object_representation.convert(state.agent.grid_object)


class AgentIDGridStateRepresentation(ArrayStateRepresentation):
    @property
    def space(self) -> Space:
        height = self.state_space.grid_shape.height
        width = self.state_space.grid_shape.width

        if height < 0 or width < 0:
            raise ValueError(f'negative height or width ({height, width})')

        return Space.make_discrete_space(
            np.zeros((height, width), dtype=int),
            np.ones((height, width), dtype=int),
        )

    def convert(self, state: State) -> np.ndarray:
        grid_agent_position = np.zeros(state.grid.shape.as_tuple, int)
        grid_agent_position[state.agent.position.yx] = 1
        return grid_agent_position


class AgentStateRepresentation(ArrayStateRepresentation):
    @property
    def space(self) -> Space:
        # 4 (last) entries for a one-hot encoding of the orientation
        return Space.make_continuous_space(
            np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        )

    def convert(self, state: State) -> np.ndarray:
        agent_array = np.zeros(6)

        # normalized between -1 and 1
        y = (2 * state.agent.position.y - state.grid.shape.height + 1) / (
            state.grid.shape.height - 1
        )
        x = (2 * state.agent.position.x - state.grid.shape.width + 1) / (
            state.grid.shape.width - 1
        )
        i = state.agent.orientation.value

        agent_array[0] = y
        agent_array[1] = x
        agent_array[2 + i] = 1

        return agent_array


# grid-object representations


class DefaultGridObjectStateRepresentation(GridObjectStateRepresentation):
    """The default representation for a grid-object

    Simply returns the grid-object indices.  See
    :func:`gym_gridverse.representations.representation.default_grid_object_representation_space`
    and
    :func:`gym_gridverse.representations.representation.default_grid_object_convert`
    for more information.
    """

    def __init__(self, state_space: StateSpace):
        super().__init__(state_space)
        self._grid_object_types = set(self.state_space.object_types) | {
            NoneGridObject
        }
        self._grid_object_colors = set(self.state_space.colors)

    @property
    def space(self) -> Space:
        return default_grid_object_representation_space(
            self._grid_object_types,
            self._grid_object_colors,
        )

    def convert(self, grid_object: GridObject) -> np.ndarray:
        return default_grid_object_representation_convert(grid_object)


class NoOverlapGridObjectStateRepresentation(GridObjectStateRepresentation):
    """The no-overlap representation for a grid-object

    Guarantees that each channel uses separate indices.  See
    :func:`gym_gridverse.representations.representation.no_overlap_grid_object_representation_space`
    and
    :func:`gym_gridverse.representations.representation.no_overlap_grid_object_convert`
    for more information.
    """

    def __init__(self, state_space: StateSpace):
        super().__init__(state_space)
        self._grid_object_types = set(self.state_space.object_types) | {
            NoneGridObject
        }
        self._grid_object_colors = set(self.state_space.colors)

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


class CompactGridObjectStateRepresentation(GridObjectStateRepresentation):
    """The compact representation for a grid-object

    Guarantees that each channel uses separate indices, and removes empty gaps
    between indices.  See
    :func:`gym_gridverse.representations.representation.compact_grid_object_representation_space`
    and
    :func:`gym_gridverse.representations.representation.compact_grid_object_convert`
    for more information.
    """

    def __init__(self, state_space: StateSpace):
        super().__init__(state_space)

        shape: Tuple[int, ...]

        # TODO eventually fix this at the space-level
        max_type_index = self.state_space.max_type_index
        max_state_index = self.state_space.max_state_index
        max_color_index = self.state_space.max_object_color

        shape = (max_type_index + 1,)
        self._grid_object_type_map = -np.ones(shape, int)
        shape = (max_type_index + 1, max_state_index + 1)
        self._grid_object_status_map = -np.ones(shape, int)
        shape = (max_color_index + 1,)
        self._grid_object_color_map = -np.ones(shape, int)

        grid_object_types = _sorted_object_types(
            set(self.state_space.object_types) | {NoneGridObject}
        )
        grid_object_colors = _sorted_colors(self.state_space.colors)

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
