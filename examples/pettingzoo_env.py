import gymnasium
import pettingzoo
import pufferlib.emulation

class SamplePettingzooEnv(pettingzoo.ParallelEnv):
    def __init__(self):
        self.possible_agents = ['agent_0', 'agent_1']
        self.agents = ['agent_0', 'agent_1']

    def observation_space(self, agent):
        return gymnasium.spaces.Box(low=-1, high=1, shape=(1,))

    def action_space(self, agent):
        return gymnasium.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        observations = {agent: self.observation_space(agent).sample() for agent in self.agents}
        return observations, {}

    def step(self, action):
        observations = {agent: self.observation_space(agent).sample() for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}
        terminals = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminals, truncations, infos

if __name__ == '__main__':
    env = SamplePettingzooEnv()
    puffer_env = pufferlib.emulation.PettingZooPufferEnv(env)
    observations, infos = puffer_env.reset()
    actions = {agent: puffer_env.action_space(agent).sample() for agent in puffer_env.agents}
    observations, rewards, terminals, truncations, infos = puffer_env.step(actions)
