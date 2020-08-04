""" Manually control the agent in an environment """

import curses
from functools import partial

import gym_gridverse.envs.observation_functions as observation_fs
import gym_gridverse.envs.reset_functions as reset_fs
import gym_gridverse.envs.reward_functions as reward_fs
import gym_gridverse.envs.state_dynamics as step_fs
import gym_gridverse.envs.terminating_functions as terminating_fs
from gym_gridverse.envs import Actions, Environment
from gym_gridverse.envs.minigrid_env import Minigrid
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid_object import Colors, Goal, GridObject
from gym_gridverse.observation import Observation
from gym_gridverse.state import State

key_action_mapping = {
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
}

agent_orientation_char_mapping = {
    Orientation.N: '▲',
    Orientation.S: '▼',
    Orientation.E: '▶',
    Orientation.W: '◀',
}


def get_action(screen) -> Actions:
    while True:
        key = screen.getch()
        try:
            return key_action_mapping[key]
        except KeyError:
            pass


def draw_object(window, position: Position, obj: GridObject):
    attribute = curses.color_pair(obj.color.value)

    try:
        # TODO change to addch if we can get newer version of ncurses
        # https://bugs.python.org/issue37738?fbclid=IwAR1Zllzw6jEG4r-1bxxoM_md8X4SabMV3NMt1nK3AqeHbyGI1iRZqDrwO6k
        window.addstr(position.y, position.x, obj.render_as_char(), attribute)
    except curses.error:
        pass


def draw_state(window, state: State):
    for position in state.grid.positions():
        draw_object(window, position, state.grid[position])

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
        window.addstr(6, 3, agent_orientation_char_mapping[Orientation.N])
    except curses.error:
        pass


def main(
    screen, env: Environment
):  # pylint: disable=too-many-locals,too-many-statements
    curses.curs_set(False)

    curses.init_pair(Colors.RED.value, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(Colors.GREEN.value, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(Colors.BLUE.value, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(
        Colors.YELLOW.value, curses.COLOR_YELLOW, curses.COLOR_BLACK
    )

    if not curses.has_colors():
        raise Exception()

    screen_height, screen_width = screen.getmaxyx()
    main_window = curses.newwin(screen_height, screen_width)

    panel_height, panel_width = 2, 22
    panel_window_outer = main_window.derwin(
        panel_height + 2, panel_width + 2, 0, 0
    )
    panel_window_inner = panel_window_outer.derwin(
        panel_height, panel_width, 1, 1
    )

    env.reset()  # TODO normally I would get grid shape from state space
    state_height, state_width = env.state.grid.shape
    state_window_outer = main_window.derwin(
        state_height + 2, state_width + 2, panel_height + 2, 0
    )
    state_window_inner = state_window_outer.derwin(
        state_height, state_width, 1, 1
    )

    observation_height, observation_width = env.observation.grid.shape
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
        screen_height - 2, screen_width - panel_width - 2 - 2, 1, 1,
    )

    action: Actions = None  # type: ignore
    reward: float = None  # type: ignore

    env.reset()
    while True:
        screen.clear()
        main_window.clear()

        # draw panel
        panel_window_outer.border()
        panel_window_outer.addstr(0, 1, 'panel')

        if action is None:
            panel_window_inner.addstr(0, 0, 'Action:')
        else:
            panel_window_inner.addstr(0, 0, f'Action: {action.name}')

        if reward is None:
            panel_window_inner.addstr(1, 0, 'Reward:')
        else:
            panel_window_inner.addstr(1, 0, f'Reward: {reward: .5g}')

        # draw state window
        state_window_outer.border()
        state_window_outer.addstr(0, 1, 'state')
        draw_state(state_window_inner, env.state)

        # draw observation window
        observation_window_outer.border()
        observation_window_outer.addstr(0, 1, 'obs.')
        draw_observation(observation_window_inner, env.observation)

        # draw legend
        legend_window_outer.border()
        legend_window_outer.addstr(0, 1, 'legend')

        def fstr(label: str, action: Actions):
            return f'{label:>8s} : {action.name}'

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

        # refresh all windows
        screen.refresh()
        main_window.refresh()

        # agent + environment step
        action = get_action(screen)
        reward, done = env.step(action)

        if done:
            env.reset()
            action = None  # type: ignore
            reward = None  # type: ignore


if __name__ == "__main__":
    reset_function = partial(reset_fs.reset_minigrid_four_rooms, 10, 10)

    step_function = step_fs.update_agent

    observation_function = observation_fs.minigrid_observation

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

    domain: Environment = Minigrid(
        reset_function,
        step_function,
        observation_function,
        reward_function,
        terminating_function,
    )

    curses.wrapper(main, domain)
