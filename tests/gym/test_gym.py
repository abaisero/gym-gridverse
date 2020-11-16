import gym


def test_gym_registration():
    gym.make('GridVerse-MiniGrid-Empty-5x5-v0')


def test_gym_control_loop():
    env = gym.make('GridVerse-MiniGrid-Empty-5x5-v0')

    env.reset()  # pylint: disable=unused-variable
    for _ in range(10):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

        if done:
            env.reset()
