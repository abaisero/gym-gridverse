from typing import Iterable, Sequence, Tuple, Type

from gym_gridverse.action import Action
from gym_gridverse.geometry import Area, Orientation, Position, Shape
from gym_gridverse.grid_object import Color, GridObject, Hidden, NoneGridObject
from gym_gridverse.observation import Observation
from gym_gridverse.state import State


def _max_object_type(object_types: Iterable[Type[GridObject]]) -> int:
    """Returns the highest object type of the provided object classes

    Args:
        object_types (`Iterable[Type[GridObject]]`):

    Returns:
        int:
    """
    return max(obj_type.type_index() for obj_type in object_types)


def _max_object_status(object_types: Iterable[Type[GridObject]]) -> int:
    """Returns the highest object status of the provided object classes

    Args:
        object_types (`Iterable[Type[GridObject]]`):

    Returns:
        int:
    """
    return max(obj_type.num_states() for obj_type in object_types)


def _max_color_index(colors: Iterable[Color]) -> int:
    """Returns the highest color index of the provided colors

    Args:
        colors (`Iterable[Color]`):

    Returns:
        int:
    """
    return max(color.value for color in colors)


class StateSpace:
    def __init__(
        self,
        grid_shape: Shape,
        object_types: Sequence[Type[GridObject]],
        colors: Sequence[Color],
    ):
        self.grid_shape = grid_shape
        self.object_types = list(object_types)
        self.colors = set(colors) | {Color.NONE}

        self._agent_object_types = set(object_types) | {NoneGridObject}

    def contains(self, state: State) -> bool:
        """True if the state satisfies the state-space"""
        # TODO: test
        return (
            state.grid.shape == self.grid_shape
            and state.grid.object_types().issubset(self.object_types)
            and state.grid.area.contains(state.agent.position)
            and isinstance(state.agent.orientation, Orientation)
            and type(state.agent.grid_object) in self._agent_object_types
        )

    @property
    def can_be_represented(self):
        # TODO: test
        return all(
            object_type.can_be_represented_in_state()
            for object_type in self.object_types
        )

    @property
    def agent_state_size(self) -> Tuple[int, int, int, int, int]:
        # TODO: test
        return (
            self.grid_shape.height,
            self.grid_shape.height,
            self.max_agent_object_type,
            self.max_agent_object_status,
            self.max_object_color,
        )

    @property
    def agent_state_shape(self) -> int:
        # TODO: test
        return len(self.agent_state_size)

    @property
    def grid_state_shape(self) -> Shape:
        # TODO: test
        return self.grid_shape

    @property
    def max_object_color(self) -> int:
        return _max_color_index(self.colors)

    # Random getters you might be interested in
    @property
    def max_type_index(self) -> int:
        return max(self.max_grid_object_type, self.max_agent_object_type)

    @property
    def max_state_index(self) -> int:
        return max(self.max_grid_object_status, self.max_agent_object_status)

    @property
    def max_grid_object_type(self) -> int:
        return _max_object_type(self.object_types)

    @property
    def max_grid_object_status(self) -> int:
        return _max_object_status(self.object_types)

    @property
    def max_agent_object_type(self) -> int:
        # NOTE: Add Hidden as the default 'non' object the agent is holding
        return _max_object_type(self.object_types + [NoneGridObject])

    @property
    def max_agent_object_status(self) -> int:
        # TODO: test
        # NOTE: Add Hidden as the default 'non' object the agent is holding
        return _max_object_status(self.object_types + [NoneGridObject])


class ActionSpace:
    def __init__(self, actions: Sequence[Action]):
        self.actions = actions

    def contains(self, action: Action) -> bool:
        """True if the action satisfies the action-space"""
        return action in self.actions

    def int_to_action(self, action: int) -> Action:
        return self.actions[action]

    def action_to_int(self, action: Action) -> int:
        # TODO: test
        return self.actions.index(action)

    @property
    def num_actions(self) -> int:
        return len(self.actions)


class ObservationSpace:
    def __init__(
        self,
        grid_shape: Shape,
        object_types: Sequence[Type[GridObject]],
        colors: Sequence[Color],
    ):
        # TODO we should generalize this
        if grid_shape.width % 2 == 0:
            raise ValueError('shape should have an odd width')

        self.grid_shape = grid_shape
        self.object_types = list(object_types)
        self.colors = set(colors) | {Color.NONE}

        self._grid_object_types = set(object_types) | {Hidden}
        self._agent_object_types = set(object_types) | {NoneGridObject}

        # TODO: eventually let this substitute the `grid_shape` input altogether
        # this area represents the observable area, with (0, 0) representing
        # the agent's position, when the agent is pointing N
        self.area = Area(
            (-self.grid_shape.height + 1, 0),
            (-(self.grid_shape.width // 2), self.grid_shape.width // 2),
        )

        # NOTE this position is relative to the top right coordinate of the area
        self.agent_position = Position(
            self.area.height - 1, self.area.width // 2
        )

        # TODO: We don't need to make assumptions about the agent position

    def contains(self, observation: Observation) -> bool:
        """True if the observation satisfies the observation-space"""
        have_same_shape = observation.grid.shape == self.grid_shape
        y_in_grid = 0 <= observation.agent.position.y < self.area.height
        x_in_grid = 0 <= observation.agent.position.x < self.area.width
        agent_obj_type_in_space = (
            type(observation.agent.grid_object) in self._agent_object_types
        )
        grid_objs_in_space = observation.grid.object_types().issubset(
            self._grid_object_types
        )
        grid_objs_colors_in_space = set(
            observation.grid[pos].color
            for pos in observation.grid.area.positions()
        ).issubset(self.colors)
        agent_obj_color_in_space = (
            observation.agent.grid_object.color in self.colors
        )

        res = [
            have_same_shape,
            grid_objs_in_space,
            grid_objs_colors_in_space,
            y_in_grid,
            x_in_grid,
            agent_obj_type_in_space,
            agent_obj_color_in_space,
        ]

        return all(res)

    @property
    def agent_state_size(self) -> Tuple[int, int, int, int, int]:
        # TODO: test
        return (
            self.grid_shape.height,
            self.grid_shape.width,
            self.max_agent_object_type,
            self.max_agent_object_status,
            self.max_object_color,
        )

    @property
    def agent_state_shape(self) -> int:
        # TODO: test
        return len(self.agent_state_size)

    @property
    def grid_state_shape(self) -> Shape:
        # TODO: test
        return self.grid_shape

    @property
    def max_object_color(self) -> int:
        return _max_color_index(self.colors)

    # Random getters you might be interested in
    @property
    def max_type_index(self) -> int:
        return max(self.max_grid_object_type, self.max_agent_object_type)

    @property
    def max_state_index(self) -> int:
        return max(self.max_grid_object_status, self.max_agent_object_status)

    @property
    def max_grid_object_type(self) -> int:
        # NOTE: Add Hidden as a potential object in any domain observation
        return _max_object_type(self.object_types + [Hidden])

    @property
    def max_grid_object_status(self) -> int:
        # NOTE: Add Hidden as a potential object in any domain observation
        return _max_object_status(self.object_types + [Hidden])

    @property
    def max_agent_object_type(self) -> int:
        # TODO: test
        # NOTE: Add Hidden as the default 'non' object the agent is holding
        return _max_object_type(self.object_types + [NoneGridObject])

    @property
    def max_agent_object_status(self) -> int:
        # TODO: test
        # NOTE: Add Hidden as the default 'non' object the agent is holding
        return _max_object_status(self.object_types + [NoneGridObject])
