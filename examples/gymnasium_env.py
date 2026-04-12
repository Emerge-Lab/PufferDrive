import gymnasium
import pufferlib.emulation

class SampleGymnasiumEnv(gymnasium.Env):
    def __init__(self):
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
        self.action_space = gymnasium.spaces.Discrete(2)

    def reset(self):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

if __name__ == '__main__':
    gymnasium_env = SampleGymnasiumEnv()
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(gymnasium_env)
    observation, info = puffer_env.reset()
    action = puffer_env.action_space.sample()
    observation, reward, terminal, truncation, info = puffer_env.step(action)
