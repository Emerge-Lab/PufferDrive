import numpy as np

import opengigaflow.spaces

ENV_ERROR = """
Environment missing required attribute {}. The most common cause is
calling super() before you have assigned the attribute.
"""
MAX_C_SEED = 2**31 - 1


def set_buffers(env, buf=None):
    if buf is None:
        obs_space = env.single_observation_space
        atn_space = env.single_action_space
        env.observations = np.zeros((env.num_agents, *obs_space.shape), dtype=obs_space.dtype)
        env.rewards = np.zeros(env.num_agents, dtype=np.float32)
        env.terminals = np.zeros(env.num_agents, dtype=bool)
        env.truncations = np.zeros(env.num_agents, dtype=bool)
        env.masks = np.ones(env.num_agents, dtype=bool)
        env.actions = np.zeros((env.num_agents, *atn_space.shape), dtype=atn_space.dtype)
    else:
        env.observations = buf["observations"]
        env.rewards = buf["rewards"]
        env.terminals = buf["terminals"]
        env.truncations = buf["truncations"]
        env.masks = buf["masks"]
        env.actions = buf["actions"]


class VecEnv:
    def __init__(self, buf=None):
        if not hasattr(self, "single_observation_space"):
            raise APIUsageError(ENV_ERROR.format("single_observation_space"))
        if not hasattr(self, "single_action_space"):
            raise APIUsageError(ENV_ERROR.format("single_action_space"))
        if not hasattr(self, "num_agents"):
            raise APIUsageError(ENV_ERROR.format("num_agents"))
        if self.num_agents < 1:
            raise APIUsageError("num_agents must be >= 1")

        if hasattr(self, "action_space"):
            raise APIUsageError("VecEnvs must define single_action_space, not action_space")
        if not isinstance(self.single_observation_space, opengigaflow.spaces.Box):
            raise APIUsageError("single_observation_space must be a Box")
        if not isinstance(self.single_action_space, opengigaflow.spaces.Box):
            raise APIUsageError("single_action_space must be a Box")

        set_buffers(self, buf)

        self.action_space = opengigaflow.spaces.joint_space(self.single_action_space, self.num_agents)

    def reset(self, seed=None):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class APIUsageError(RuntimeError):
    def __init__(self, message="API usage error."):
        self.message = message
        super().__init__(self.message)


def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v
