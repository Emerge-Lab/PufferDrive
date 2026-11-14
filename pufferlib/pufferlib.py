import numpy as np

import pufferlib.spaces

ENV_ERROR = """
Environment missing required attribute {}. The most common cause is
calling super() before you have assigned the attribute.
"""


def set_buffers(env, buf=None):
    if buf is None:
        obs_space = env.single_observation_space
        env.observations = np.zeros((env.num_agents, *obs_space.shape), dtype=obs_space.dtype)
        env.rewards = np.zeros(env.num_agents, dtype=np.float32)
        env.terminals = np.zeros(env.num_agents, dtype=bool)
        env.truncations = np.zeros(env.num_agents, dtype=bool)
        env.masks = np.ones(env.num_agents, dtype=bool)

        # TODO: Major kerfuffle on inferring action space dtype. This needs some asserts?
        atn_space = pufferlib.spaces.joint_space(env.single_action_space, env.num_agents)
        if isinstance(env.single_action_space, pufferlib.spaces.Box):
            env.actions = np.zeros(atn_space.shape, dtype=atn_space.dtype)
        else:
            env.actions = np.zeros(atn_space.shape, dtype=np.int32)
    else:
        env.observations = buf["observations"]
        env.rewards = buf["rewards"]
        env.terminals = buf["terminals"]
        env.truncations = buf["truncations"]
        env.masks = buf["masks"]
        env.actions = buf["actions"]


class PufferEnv:
    def __init__(self, buf=None):
        if not hasattr(self, "single_observation_space"):
            raise APIUsageError(ENV_ERROR.format("single_observation_space"))
        if not hasattr(self, "single_action_space"):
            raise APIUsageError(ENV_ERROR.format("single_action_space"))
        if not hasattr(self, "num_agents"):
            raise APIUsageError(ENV_ERROR.format("num_agents"))
        if self.num_agents < 1:
            raise APIUsageError("num_agents must be >= 1")

        if hasattr(self, "observation_space"):
            raise APIUsageError("PufferEnvs must define single_observation_space, not observation_space")
        if hasattr(self, "action_space"):
            raise APIUsageError("PufferEnvs must define single_action_space, not action_space")
        if not isinstance(self.single_observation_space, pufferlib.spaces.Box):
            raise APIUsageError("Native observation_space must be a Box")
        if (
            not isinstance(self.single_action_space, pufferlib.spaces.Discrete)
            and not isinstance(self.single_action_space, pufferlib.spaces.MultiDiscrete)
            and not isinstance(self.single_action_space, pufferlib.spaces.Box)
        ):
            raise APIUsageError("Native action_space must be a Discrete, MultiDiscrete, or Box")

        set_buffers(self, buf)

        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.observation_space = pufferlib.spaces.joint_space(self.single_observation_space, self.num_agents)
        self.agent_ids = np.arange(self.num_agents)

    @property
    def emulated(self):
        """Native envs do not use emulation"""
        return False

    @property
    def done(self):
        """Native envs handle resets internally"""
        return False

    @property
    def driver_env(self):
        """For compatibility with Multiprocessing"""
        return self

    def reset(self, seed=None):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def async_reset(self, seed=None):
        _, self.infos = self.reset(seed)
        assert isinstance(self.infos, list), "PufferEnvs must return info as a list of dicts"

    def send(self, actions):
        _, _, _, _, self.infos = self.step(actions)
        assert isinstance(self.infos, list), "PufferEnvs must return info as a list of dicts"

    def recv(self):
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            self.infos,
            self.agent_ids,
            self.masks,
        )


### Exceptions
class APIUsageError(RuntimeError):
    """Exception raised when the API is used incorrectly."""

    def __init__(self, message="API usage error."):
        self.message = message
        super().__init__(self.message)


### Misc
def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v
