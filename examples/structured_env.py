import gymnasium
import pufferlib.emulation

class SampleGymnasiumEnv(gymnasium.Env):
    def __init__(self):
        self.observation_space = gymnasium.spaces.Dict({
            'foo': gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
            'bar': gymnasium.spaces.Box(low=2, high=3, shape=(3,)),
        })
        self.action_space = gymnasium.spaces.MultiDiscrete([2, 5])

    def reset(self):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

if __name__ == '__main__':
    gymnasium_env = SampleGymnasiumEnv()
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(gymnasium_env)
    flat_observation, info = puffer_env.reset()
    flat_action = puffer_env.action_space.sample()
    flat_observation, reward, terminal, truncation, info = puffer_env.step(flat_action)
    print(f'PufferLib flattens observations and actions:\n{flat_observation}\n{flat_action}')

    observation = flat_observation.view(puffer_env.obs_dtype)
    print(f'You can unflatten observations with numpy:\n{observation}')

    import torch
    import pufferlib.pytorch
    flat_torch_observation = torch.from_numpy(flat_observation)
    torch_dtype = pufferlib.pytorch.nativize_dtype(puffer_env.emulated)
    torch_observation = pufferlib.pytorch.nativize_tensor(flat_torch_observation, torch_dtype)
    print(f'But we suggest unflattening observations with torch in your model forward pass:\n{torch_observation}')
