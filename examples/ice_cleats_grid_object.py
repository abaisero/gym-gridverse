from gym_gridverse.grid_object import Color, GridObject


class IceCleats(GridObject):
    state_index = 0
    color = Color.NONE
    blocks_movement = False
    blocks_vision = False
    blocks_holdable = True

    @classmethod
    def can_be_represented_in_state(cls) -> bool:
        return True

    @classmethod
    def num_states(cls) -> int:
        return 1
