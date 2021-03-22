import numpy as np
import pytest

from gym_gridverse.agent import Agent
from gym_gridverse.envs.reset_functions import reset_empty
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import (
    Color,
    Door,
    Floor,
    Goal,
    Key,
    NoneGridObject,
    Wall,
)
from gym_gridverse.representations.representation import (
    default_convert,
    default_representation_space,
    no_overlap_convert,
    no_overlap_representation_space,
)


@pytest.fixture
def default_representation_fixture():
    """Creates an observation representation of 3 by 3"""
    height, width = 3, 3

    max_obj_type = Key.type_index  # pylint: disable=no-member
    max_obj_state = Door.num_states()
    max_color_value = Color.BLUE.value

    return height, width, max_obj_type, max_obj_state, max_color_value


def test_default_representation_space(
    default_representation_fixture,
):  # pylint: disable = redefined-outer-name
    (
        height,
        width,
        max_obj_type,
        max_obj_state,
        max_color_value,
    ) = default_representation_fixture

    expected_item_space = [max_obj_type, max_obj_state, max_color_value]
    expected_grid_space = np.array([[expected_item_space] * width] * height)

    space = default_representation_space(
        max_obj_type, max_obj_state, max_color_value, width, height
    )

    np.testing.assert_array_equal(
        space['grid'].upper_bound, expected_grid_space
    )
    np.testing.assert_array_equal(
        space['grid'].lower_bound, np.zeros((height, width, 3), dtype=int)
    )
    np.testing.assert_equal(
        space['agent_id_grid'].upper_bound, np.ones((height, width), dtype=int)
    )
    np.testing.assert_equal(
        space['agent_id_grid'].lower_bound, np.zeros((height, width), dtype=int)
    )
    np.testing.assert_array_equal(
        space['item'].upper_bound, expected_item_space
    )
    np.testing.assert_array_equal(space['item'].lower_bound, [0, 0, 0])
    np.testing.assert_array_equal(
        space['agent'].upper_bound, [height - 1, width - 1, 1, 1, 1, 1]
    )
    np.testing.assert_array_equal(
        space['agent'].lower_bound, np.zeros(6, dtype=int)
    )

    # legacy
    expected_legacy_grid_space = np.array(
        [[expected_item_space * 2] * width] * height
    )
    np.testing.assert_array_equal(
        space['legacy-grid'].upper_bound, expected_legacy_grid_space
    )
    np.testing.assert_array_equal(
        space['legacy-grid'].lower_bound,
        np.zeros((height, width, 6), dtype=int),
    )
    np.testing.assert_array_equal(
        space['legacy-agent'].upper_bound,
        [
            height - 1,
            width - 1,
            len(Orientation) - 1,
            max_obj_type,
            max_obj_state,
            max_color_value,
        ],
    )
    np.testing.assert_array_equal(
        space['legacy-agent'].lower_bound,
        np.zeros(6, dtype=int),
    )


def test_default_representation_convert(
    default_representation_fixture,
):  # pylint: disable = redefined-outer-name
    height, width, _, _, _ = default_representation_fixture

    agent = Agent(Position(0, 2), Orientation.N, Key(Color.RED))
    grid = Grid(height, width)
    grid[1, 1] = Door(Door.Status.CLOSED, Color.BLUE)

    floor_index = Floor.type_index  # pylint: disable=no-member

    # y, x, one-hot encoding with North (== 0)
    expected_agent_representation = [0, 2, 1, 0, 0, 0]
    expected_item_representation = [
        agent.obj.type_index,
        agent.obj.state_index,
        agent.obj.color.value,
    ]
    expected_legacy_agent_representation = np.array(
        [
            0,
            2,
            Orientation.N.value,
            agent.obj.type_index,
            agent.obj.state_index,
            agent.obj.color.value,
        ]
    )

    expected_agent_position_channels = np.zeros((3, 3, 3))
    expected_agent_position_channels[:, :, 0] = NoneGridObject.type_index
    expected_agent_position_channels[:, :, 1] = 0  # status
    expected_agent_position_channels[:, :, 2] = Color.NONE.value

    expected_agent_position_channels[
        0, 2
    ] = expected_legacy_agent_representation[3:]

    expected_grid_representation = np.array(
        [
            [
                [floor_index, 0, 0],
                [floor_index, 0, 0],
                [
                    floor_index,
                    0,
                    0,
                ],
            ],
            [
                [floor_index, 0, 0],
                [
                    Door.type_index,
                    Door.Status.CLOSED.value,
                    Color.BLUE.value,
                ],
                [floor_index, 0, 0],
            ],
            [
                [floor_index, 0, 0],
                [floor_index, 0, 0],
                [floor_index, 0, 0],
            ],
        ]
    )
    expected_agent_id_grid = np.zeros((height, width), dtype=int)
    expected_agent_id_grid[0, 2] = 1

    rep = default_convert(grid, agent)

    np.testing.assert_array_equal(rep['grid'], expected_grid_representation)
    np.testing.assert_array_equal(rep['agent_id_grid'], expected_agent_id_grid)
    np.testing.assert_array_equal(rep['agent'], expected_agent_representation)
    np.testing.assert_array_equal(rep['item'], expected_item_representation)

    # legacy
    np.testing.assert_array_equal(
        rep['legacy-grid'][:, :, :3], expected_agent_position_channels
    )
    np.testing.assert_array_equal(
        rep['legacy-grid'][:, :, 3:], expected_grid_representation
    )
    np.testing.assert_array_equal(
        rep['legacy-agent'], expected_legacy_agent_representation
    )


@pytest.fixture
def no_overlap_fixture():
    """Creates a state representation of 4 by 5"""
    height, width = 4, 5

    # hard coded from above
    max_object_type = Goal.type_index  # pylint: disable=no-member
    max_object_status = 0
    max_color_value = Color.GREEN.value

    return height, width, max_object_type, max_object_status, max_color_value


def test_no_overlap_space(
    no_overlap_fixture,
):  # pylint: disable = redefined-outer-name
    (
        height,
        width,
        max_object_type,
        max_object_status,
        max_color_value,
    ) = no_overlap_fixture

    max_channel_values = [
        max_object_type,
        max_object_type + max_object_status + 1,
        max_object_type + max_object_status + max_color_value + 2,
    ]

    expected_grid_object_space = np.array(
        [[max_channel_values] * width] * height
    )

    space = no_overlap_representation_space(
        max_object_type, max_object_status, max_color_value, width, height
    )

    np.testing.assert_array_equal(
        space['grid'].upper_bound, expected_grid_object_space
    )
    np.testing.assert_array_equal(
        space['grid'].lower_bound, np.zeros((height, width, 3), dtype=int)
    )
    np.testing.assert_equal(
        space['agent_id_grid'].upper_bound, np.ones((height, width), dtype=int)
    )
    np.testing.assert_equal(
        space['agent_id_grid'].lower_bound, np.zeros((height, width), dtype=int)
    )
    np.testing.assert_array_equal(space['item'].upper_bound, max_channel_values)
    np.testing.assert_array_equal(space['item'].lower_bound, [0, 0, 0])

    np.testing.assert_equal(
        space['agent'].upper_bound, [height - 1, width - 1, 1, 1, 1, 1]
    )
    np.testing.assert_equal(space['agent'].lower_bound, np.zeros(6, dtype=int))

    # legacy
    expected_legacy_grid_space = np.array(
        [[max_channel_values * 2] * width] * height
    )

    np.testing.assert_array_equal(
        space['legacy-grid'].upper_bound, expected_legacy_grid_space
    )
    np.testing.assert_array_equal(
        space['legacy-grid'].lower_bound,
        np.zeros((height, width, 6), dtype=int),
    )
    np.testing.assert_array_equal(
        space['legacy-agent'].upper_bound, max_channel_values
    )
    np.testing.assert_array_equal(
        space['legacy-agent'].lower_bound, np.zeros(3, dtype=int)
    )


def test_no_overlap_convert(
    no_overlap_fixture,
):  # pylint: disable = redefined-outer-name
    height, width, max_object_type, max_object_status, _ = no_overlap_fixture
    first_item_status = max_object_type + 1
    first_item_color = max_object_type + max_object_status + 2

    state = reset_empty(height, width, random_agent=True)

    # pylint: disable=no-member
    expected_agent_state = np.array(
        [
            NoneGridObject.type_index,
            first_item_status,
            first_item_color,
        ]
    )

    expected_agent_position_channels = np.zeros((height, width, 3))
    expected_agent_position_channels[:, :] = [
        NoneGridObject.type_index,
        first_item_status,
        first_item_color,
    ]

    expected_agent_position_channels[
        state.agent.position.astuple()
    ] = expected_agent_state

    expected_grid_state = np.array(
        [
            [
                [
                    Floor.type_index,
                    first_item_status,
                    first_item_color,
                ]
            ]
            * width
        ]
        * height
    )

    # we expect walls to be around
    expected_grid_state[0, :] = [
        Wall.type_index,
        first_item_status,
        first_item_color,
    ]
    expected_grid_state[height - 1, :] = [
        Wall.type_index,
        first_item_status,
        first_item_color,
    ]
    expected_grid_state[:, 0] = [
        Wall.type_index,
        first_item_status,
        first_item_color,
    ]
    expected_grid_state[:, width - 1] = [
        Wall.type_index,
        first_item_status,
        first_item_color,
    ]

    # we expect goal to be in corner
    expected_grid_state[height - 2, width - 2, :] = [
        Goal.type_index,
        first_item_status,
        first_item_color,
    ]

    representation = no_overlap_convert(
        state.grid, state.agent, max_object_type, max_object_status
    )

    expected_agent_id_grid = np.zeros((height, width), dtype=int)
    expected_agent_id_grid[state.agent.position.astuple()] = 1

    np.testing.assert_array_equal(representation['grid'], expected_grid_state)
    np.testing.assert_array_equal(
        representation['agent_id_grid'], expected_agent_id_grid
    )
    np.testing.assert_array_equal(representation['item'], expected_agent_state)

    expected_agent_representation = np.zeros(6, dtype=int)
    expected_agent_representation[0] = state.agent.position.y
    expected_agent_representation[1] = state.agent.position.x
    expected_agent_representation[2 + state.agent.orientation.value] = 1
    np.testing.assert_equal(
        representation['agent'], expected_agent_representation
    )

    # legacy
    np.testing.assert_array_equal(
        representation['legacy-grid'][:, :, :3],
        expected_agent_position_channels,
    )
    np.testing.assert_array_equal(
        representation['legacy-grid'][:, :, 3:], expected_grid_state
    )
    np.testing.assert_array_equal(
        representation['legacy-agent'], expected_agent_state
    )
