""" Manually control the agent in an environment """
from __future__ import annotations

import curses
import enum
from dataclasses import dataclass
from functools import partial
from typing import Optional

import gym_gridverse.envs.observation_functions as observation_fs
import gym_gridverse.envs.reset_functions as reset_fs
import gym_gridverse.envs.reward_functions as reward_fs
import gym_gridverse.envs.terminating_functions as terminating_fs
import gym_gridverse.envs.transition_functions as step_fs
from gym_gridverse.actions import Actions
from gym_gridverse.envs import InnerEnv
from gym_gridverse.envs.gridworld import GridWorld
from gym_gridverse.geometry import Orientation, Position, Shape
from gym_gridverse.grid_object import Colors, Floor, Goal, GridObject, Wall
from gym_gridverse.observation import Observation
from gym_gridverse.spaces import (
    ActionSpace,
    DomainSpace,
    ObservationSpace,
    StateSpace,
)
from gym_gridverse.state import State


class Controls(enum.Enum):
    QUIT = enum.auto()
    RESET = enum.auto()
    HIDE_STATE = enum.auto()


key_value_mapping = {
    # actions
    curses.KEY_UP: Actions.MOVE_FORWARD,
    curses.KEY_DOWN: Actions.MOVE_BACKWARD,
    curses.KEY_LEFT: Actions.TURN_LEFT,
    curses.KEY_RIGHT: Actions.TURN_RIGHT,
    ord('w'): Actions.MOVE_FORWARD,
    ord('a'): Actions.MOVE_LEFT,
    ord('s'): Actions.MOVE_BACKWARD,
    ord('d'): Actions.MOVE_RIGHT,
    ord(' '): Actions.ACTUATE,
    ord('p'): Actions.PICK_N_DROP,
    # controls
    ord('q'): Controls.QUIT,
    ord('r'): Controls.RESET,
    ord('h'): Controls.HIDE_STATE,
}

agent_orientation_char_mapping = {
    Orientation.N: '▲',
    Orientation.S: '▼',
    Orientation.E: '▶',
    Orientation.W: '◀',
}


def draw_object(
    window, position: Position, obj: GridObject, *, hidden: bool = False
):
    attribute = curses.color_pair(obj.color.value)

    if hidden:
        attribute |= curses.A_REVERSE

    try:
        # TODO change to addch if we can get newer version of ncurses
        # https://bugs.python.org/issue37738?fbclid=IwAR1Zllzw6jEG4r-1bxxoM_md8X4SabMV3NMt1nK3AqeHbyGI1iRZqDrwO6k
        window.addstr(position.y, position.x, obj.render_as_char(), attribute)
    except curses.error:
        pass


def draw_state(window, state: State, observation_space: ObservationSpace):
    area = state.agent.get_pov_area(observation_space.area)
    for position in state.grid.positions():
        draw_object(
            window,
            position,
            state.grid[position],
            hidden=not area.contains(position),
        )

    try:
        # TODO change to addch if we can get newer version of ncurses
        # https://bugs.python.org/issue37738?fbclid=IwAR1Zllzw6jEG4r-1bxxoM_md8X4SabMV3NMt1nK3AqeHbyGI1iRZqDrwO6k
        window.addstr(
            state.agent.position.y,
            state.agent.position.x,
            agent_orientation_char_mapping[state.agent.orientation],
        )
    except curses.error:
        pass


def draw_observation(window, observation: Observation):
    for position in observation.grid.positions():
        draw_object(window, position, observation.grid[position])

    try:
        # TODO change to addch if we can get newer version of ncurses
        # https://bugs.python.org/issue37738?fbclid=IwAR1Zllzw6jEG4r-1bxxoM_md8X4SabMV3NMt1nK3AqeHbyGI1iRZqDrwO6k
        window.addstr(
            observation.agent.position.y,
            observation.agent.position.x,
            agent_orientation_char_mapping[Orientation.N],
        )
    except curses.error:
        pass


def main(
    screen, env: InnerEnv
):  # pylint: disable=too-many-locals,too-many-statements
    curses.curs_set(False)

    curses.init_pair(Colors.RED.value, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(Colors.GREEN.value, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(Colors.BLUE.value, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(
        Colors.YELLOW.value, curses.COLOR_YELLOW, curses.COLOR_BLACK
    )

    screen_height, screen_width = screen.getmaxyx()
    main_window = curses.newwin(screen_height, screen_width)

    panel_height, panel_width = 3, 22
    panel_window_outer = main_window.derwin(
        panel_height + 2, panel_width + 2, 0, 0
    )
    panel_window_inner = panel_window_outer.derwin(
        panel_height, panel_width, 1, 1
    )

    state_height, state_width = env.state_space.grid_shape
    state_window_outer = main_window.derwin(
        state_height + 2, state_width + 2, panel_height + 2, 0
    )
    state_window_inner = state_window_outer.derwin(
        state_height, state_width, 1, 1
    )

    observation_height, observation_width = env.observation_space.grid_shape
    observation_window_outer = main_window.derwin(
        observation_height + 2,
        observation_width + 2,
        panel_height + 2,
        state_width + 2 + 1,
    )
    observation_window_inner = observation_window_outer.derwin(
        observation_height, observation_width, 1, 1
    )

    legend_window_outer = main_window.derwin(
        screen_height,
        screen_width - panel_width - 2,
        0,
        state_width + 2 + observation_width + 2 + 3,
    )
    legend_window_inner = legend_window_outer.derwin(
        screen_height - 2,
        screen_width - panel_width - 2 - 2,
        1,
        1,
    )

    def update(
        viz_state: VizState, observation_space: ObservationSpace
    ):  # pylint: disable=too-many-arguments
        screen.clear()
        main_window.clear()

        # draw panel
        panel_window_outer.border()
        panel_window_outer.addstr(0, 1, 'panel')

        panel_window_inner.addstr(0, 0, f'Step: {viz_state.t}')
        if viz_state.action is None:
            panel_window_inner.addstr(1, 0, 'Action:')
        else:
            panel_window_inner.addstr(1, 0, f'Action: {viz_state.action.name}')

        if viz_state.reward is None:
            panel_window_inner.addstr(2, 0, 'Reward:')
        else:
            panel_window_inner.addstr(2, 0, f'Reward: {viz_state.reward: .5g}')

        # draw state window
        state_window_outer.border()
        state_window_outer.addstr(0, 1, 'state')
        if not viz_state.hide_state:
            draw_state(state_window_inner, viz_state.state, observation_space)

        # draw observation window
        observation_window_outer.border()
        observation_window_outer.addstr(0, 1, 'obs.')
        draw_observation(observation_window_inner, viz_state.observation)

        # draw legend
        legend_window_outer.border()
        legend_window_outer.addstr(0, 1, 'legend')

        def fstr(label: str, e: enum.Enum):
            return f'{label:>8s} : {e.name}'

        legend_window_inner.addstr(0, 0, fstr('<UP>', Actions.MOVE_FORWARD))
        legend_window_inner.addstr(1, 0, fstr('<DOWN>', Actions.MOVE_BACKWARD))
        legend_window_inner.addstr(2, 0, fstr('<LEFT>', Actions.TURN_LEFT))
        legend_window_inner.addstr(3, 0, fstr('<RIGHT>', Actions.TURN_RIGHT))

        legend_window_inner.addstr(5, 0, fstr('w', Actions.MOVE_FORWARD))
        legend_window_inner.addstr(6, 0, fstr('a', Actions.MOVE_LEFT))
        legend_window_inner.addstr(7, 0, fstr('s', Actions.MOVE_BACKWARD))
        legend_window_inner.addstr(8, 0, fstr('d', Actions.MOVE_RIGHT))

        legend_window_inner.addstr(10, 0, fstr('<SPACE>', Actions.ACTUATE))
        legend_window_inner.addstr(11, 0, fstr('p', Actions.PICK_N_DROP))

        legend_window_inner.addstr(13, 0, fstr('q', Controls.QUIT))
        legend_window_inner.addstr(14, 0, fstr('r', Controls.RESET))
        legend_window_inner.addstr(15, 0, fstr('h', Controls.HIDE_STATE))

        # refresh all windows
        screen.refresh()
        main_window.refresh()
        legend_window_outer.refresh()

    @dataclass
    class VizState:
        t: int
        action: Optional[Actions]
        reward: Optional[float]
        state: State
        observation: Observation
        hide_state: bool

        @classmethod
        def from_env(cls, env: InnerEnv, *, hide_state: bool) -> VizState:
            state = env.functional_reset()
            observation = env.functional_observation(state)
            return VizState(
                t=0,
                action=None,
                reward=None,
                state=state,
                observation=observation,
                hide_state=hide_state,
            )

    viz_state = VizState.from_env(env, hide_state=False)
    while True:
        update(viz_state, env.observation_space)
        key = screen.getch()

        try:
            value = key_value_mapping[key]
        except KeyError:
            continue

        if isinstance(value, Actions):
            viz_state.action = value

            viz_state.t += 1
            viz_state.state, viz_state.reward, done = env.functional_step(
                viz_state.state, viz_state.action
            )
            viz_state.observation = env.functional_observation(viz_state.state)

            if done:
                update(viz_state, env.observation_space)
                screen.getch()

                viz_state = VizState.from_env(
                    env, hide_state=viz_state.hide_state
                )

        elif value == Controls.QUIT:
            break

        elif value == Controls.RESET:
            viz_state = VizState.from_env(env, hide_state=viz_state.hide_state)

        elif value == Controls.HIDE_STATE:
            viz_state.hide_state = not viz_state.hide_state


if __name__ == "__main__":
    domain_space = DomainSpace(
        StateSpace(Shape(10, 10), [Floor, Wall, Goal], [Colors.NONE]),
        ActionSpace(list(Actions)),
        ObservationSpace(Shape(7, 7), [Floor, Wall, Goal], [Colors.NONE]),
    )

    # reset_function = partial(reset_fs.reset_minigrid_rooms, 10, 10, (2, 2))
    reset_function = partial(reset_fs.reset_minigrid_empty, 10, 10, True)

    step_function = step_fs.update_agent

    observation_function = observation_fs.full_visibility
    # observation_function = observation_fs.minigrid_observation
    # observation_function = observation_fs.raytracing_observation
    # observation_function = observation_fs.stochastic_raytracing_observation
    observation_function = partial(
        observation_function,
        observation_space=domain_space.observation_space,
    )

    reward_function = partial(
        reward_fs.chain,
        reward_functions=[
            partial(reward_fs.living_reward, reward=-0.01),
            partial(reward_fs.reach_goal, reward_on=10.0),
            partial(
                reward_fs.getting_closer,
                reward_closer=0.1,
                reward_further=-0.1,
                object_type=Goal,
            ),
        ],
    )

    terminating_function = terminating_fs.reach_goal

    domain: InnerEnv = GridWorld(
        domain_space,
        reset_function,
        step_function,
        observation_function,
        reward_function,
        terminating_function,
    )

    curses.wrapper(main, domain)
