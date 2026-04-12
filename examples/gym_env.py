import gym
import pufferlib.emulation

class SampleGymEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}

if __name__ == '__main__':
    gym_env = SampleGymEnv()
    gymnasium_env = pufferlib.GymToGymnasium(gym_env)
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(gymnasium_env)
    observations, info = puffer_env.reset()
    action = puffer_env.action_space.sample()
    observation, reward, terminal, truncation, info = puffer_env.step(action)
