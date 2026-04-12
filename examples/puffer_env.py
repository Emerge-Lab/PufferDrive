import gymnasium
import pufferlib.emulation

class SamplePufferEnv(pufferlib.PufferEnv):
    def __init__(self, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.num_agents = 2
        super().__init__(buf)

    def reset(self, seed=0):
        self.observations[:] = self.observation_space.sample()
        return self.observations, []

    def step(self, action):
        self.observations[:] = self.observation_space.sample()
        infos = [{'infos': 'is a list of dictionaries'}]
        return self.observations, self.rewards, self.terminals, self.truncations, infos

if __name__ == '__main__':
    puffer_env = SamplePufferEnv()
    observations, infos = puffer_env.reset()
    actions = puffer_env.action_space.sample()
    observations, rewards, terminals, truncations, infos = puffer_env.step(actions)
    print('Puffer envs use a vector interface and in-place array updates')
    print('Observation:', observations)
    print('Reward:', rewards)
    print('Terminal:', terminals)
    print('Truncation:', truncations)
