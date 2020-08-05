from typing import List, Tuple, Type

from gym_gridverse.actions import Actions
from gym_gridverse.geometry import Shape
from gym_gridverse.grid_object import Colors, GridObject, Hidden, NoneGridObject


def _max_object_type(objects: List[Type[GridObject]]) -> int:
    """Returns the highest object type of the provided object classes

    Args:
        objects (`List[Type[GridObject]]`):

    Returns:
        int:
    """
    return max(obj_type.type_index for obj_type in objects)


def _max_object_status(objects: List[Type[GridObject]]) -> int:
    """Returns the highest object status of the provided object classes

    Args:
        objects (`List[Type[GridObject]]`):

    Returns:
        int:
    """
    return max(obj_type.num_states() for obj_type in objects)


def _max_color_index(colors: List[Colors]) -> int:
    """Returns the highest color index of the provided colors

    Args:
        objects (`List[Colors]`):

    Returns:
        int:
    """
    return max(color.value for color in colors)


class StateSpace:
    def __init__(
        self,
        grid_shape: Shape,
        objects: List[Type[GridObject]],
        colors: List[Colors],
    ):
        self.grid_shape = grid_shape
        self.objects = objects
        self.colors = colors

    @property
    def agent_state_size(self) -> Tuple[int, int, int, int, int]:
        return (
            self.grid_shape[0],
            self.grid_shape[1],
            self.max_agent_object_type,
            self.max_agent_object_status,
            self.max_object_color,
        )

    @property
    def agent_state_shape(self) -> int:
        return len(self.agent_state_size)

    @property
    def grid_state_shape(self) -> Shape:
        return self.grid_shape

    @property
    def max_object_color(self) -> int:
        return _max_color_index(self.colors)

    # Random getters you might be interested in
    @property
    def max_grid_object_type(self) -> int:
        return _max_object_type(self.objects)

    @property
    def max_grid_object_status(self) -> int:
        return _max_object_status(self.objects)

    @property
    def max_agent_object_type(self) -> int:
        # NOTE: Add Hidden as the default 'non' object the agent is holding
        return _max_object_type(self.objects + [NoneGridObject])

    @property
    def max_agent_object_status(self) -> int:
        # NOTE: Add Hidden as the default 'non' object the agent is holding
        return _max_object_status(self.objects + [NoneGridObject])


class ActionSpace:
    def __init__(self, actions: List[Actions]):
        self.actions = actions

    def int_to_action(self, action: int) -> Actions:
        return self.actions[action]

    def action_to_int(self, action: Actions) -> int:
        return self.actions.index(action)

    @property
    def num_actions(self) -> int:
        return len(self.actions)


class ObservationSpace:
    def __init__(
        self,
        grid_shape: Shape,
        objects: List[Type[GridObject]],
        colors: List[Colors],
    ):
        self.grid_shape = grid_shape
        self.objects = objects
        self.colors = colors

    @property
    def agent_state_size(self) -> Tuple[int, int, int, int, int]:
        return (
            self.grid_shape[0],
            self.grid_shape[1],
            self.max_agent_object_type,
            self.max_agent_object_status,
            self.max_object_color,
        )

    @property
    def agent_state_shape(self) -> int:
        return len(self.agent_state_size)

    @property
    def grid_state_shape(self) -> Shape:
        return self.grid_shape

    @property
    def max_object_color(self) -> int:
        return _max_color_index(self.colors)

    # Random getters you might be interested in
    @property
    def max_grid_object_type(self) -> int:
        # NOTE: Add Hidden as a potential object in any domain observation
        return _max_object_type(self.objects + [Hidden])

    @property
    def max_grid_object_status(self) -> int:
        # NOTE: Add Hidden as a potential object in any domain observation
        return _max_object_status(self.objects + [Hidden])

    @property
    def max_agent_object_type(self) -> int:
        # NOTE: Add Hidden as the default 'non' object the agent is holding
        return _max_object_type(self.objects + [NoneGridObject])

    @property
    def max_agent_object_status(self) -> int:
        # NOTE: Add Hidden as the default 'non' object the agent is holding
        return _max_object_status(self.objects + [NoneGridObject])
