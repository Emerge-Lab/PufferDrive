import numpy as np

import pufferlib
import pufferlib.emulation
import pufferlib.utils
import pufferlib.vector
from pufferlib.environments import test

# Deprecation warnings from gymnasium
import warnings

warnings.filterwarnings("ignore")


class RandomState:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def random(self):
        return self.rng.random()

    def probabilistic_round(self, n):
        frac, integer = np.modf(n)
        if self.random() < frac:
            return int(integer) + 1
        else:
            return int(integer)

    def sample(self, ary, n):
        n_rounded = self.probabilistic_round(n)
        return self.rng.choice(ary, n_rounded, replace=False).tolist()

    def choice(self, ary):
        return self.sample(ary, 1)[0]


# TODO: Fix this. Was in utils.py. Only used for tests
def make_zeros_like(data):
    if isinstance(data, dict):
        return {k: make_zeros_like(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_zeros_like(v) for v in data]
    elif isinstance(data, np.ndarray):
        return np.zeros_like(data)
    elif isinstance(data, (int, float)):
        return 0
    else:
        raise ValueError(f"Unsupported type: {type(data)}")


def compare_arrays(array_1, array_2):
    assert isinstance(array_1, np.ndarray)
    assert isinstance(array_2, np.ndarray)
    assert array_1.shape == array_2.shape
    return np.allclose(array_1, array_2)


def compare_dicts(dict_1, dict_2, idx):
    assert isinstance(dict_1, (dict, OrderedDict))
    assert isinstance(dict_2, (dict, OrderedDict))

    if not all(k in dict_2 for k in dict_1):
        raise ValueError("Keys do not match between dictionaries.")

    for k, v in dict_1.items():
        if not compare_space_samples(v, dict_2[k], idx):
            return False

    return True


def compare_lists(list_1, list_2, idx):
    assert isinstance(list_1, (list, tuple))
    assert isinstance(list_2, (list, tuple))

    if len(list_1) != len(list_2):
        raise ValueError("Lengths do not match between lists/tuples.")

    for v1, v2 in zip(list_1, list_2):
        if not compare_space_samples(v1, v2, idx):
            return False

    return True


def compare_space_samples(sample_1, sample_2, sample_2_batch_idx=None):
    """Compare two samples from the same space

    Optionally, sample_2 may be a batch of samples from the same space
    concatenated along the first dimension of the leaves. In this case,
    sample_2_batch_idx specifies which sample to compare.
    """
    if isinstance(sample_1, (dict, OrderedDict)):
        return compare_dicts(sample_1, sample_2, sample_2_batch_idx)
    elif isinstance(sample_1, (list, tuple)):
        return compare_lists(sample_1, sample_2, sample_2_batch_idx)
    elif isinstance(sample_1, np.ndarray):
        assert isinstance(sample_2, np.ndarray)
        if sample_2_batch_idx is not None:
            sample_2 = sample_2[sample_2_batch_idx]
        return compare_arrays(sample_1, sample_2)
    elif isinstance(sample_1, (int, float)):
        if sample_2_batch_idx is not None:
            sample_2 = sample_2[sample_2_batch_idx]
        if isinstance(sample_2, np.ndarray):
            assert sample_2.size == 1, "Cannot compare scalar to non-scalar."
            sample_2 = sample_2[0]
        return sample_1 == sample_2
    else:
        raise ValueError(f"Unsupported type: {type(sample_1)}")


def test_gymnasium_emulation(env_cls, steps=100):
    raw_env = env_cls()
    puf_env = pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_cls)

    raw_done = puf_done = True
    raw_truncated = puf_truncated = False

    for i in range(steps):
        assert puf_done == raw_done
        assert puf_truncated == raw_truncated

        if raw_done:
            puf_ob, _ = puf_env.reset()
            raw_ob, _ = raw_env.reset()

        # Reconstruct original obs format from puffer env and compare to raw
        if puf_env.is_obs_emulated:
            puf_ob = pufferlib.emulation.nativize(puf_ob, puf_env.env.observation_space, puf_env.obs_dtype)

        pufferlib.utils.compare_space_samples(raw_ob, puf_ob)

        action = raw_env.action_space.sample()
        raw_ob, raw_reward, raw_done, raw_truncated, _ = raw_env.step(action)

        # Convert raw actions to puffer format
        if puf_env.is_atn_emulated:
            action = pufferlib.emulation.emulate_copy(action, puf_env.action_space.dtype, puf_env.atn_dtype)

        puf_ob, puf_reward, puf_done, puf_truncated, _ = puf_env.step(action)
        assert puf_reward == raw_reward


def test_pettingzoo_emulation(env_cls, steps=100):
    raw_env = env_cls()
    puf_env = pufferlib.emulation.PettingZooPufferEnv(env_creator=env_cls)

    for i in range(steps):
        raw_done = len(raw_env.agents) == 0
        puf_done = len(puf_env.agents) == 0

        assert puf_done == raw_done

        if raw_done:
            puf_obs, _ = puf_env.reset()
            raw_obs, _ = raw_env.reset()

        for agent in puf_env.possible_agents:
            if agent not in raw_obs:
                assert np.sum(puf_obs[agent] != 0) == 0
                continue

            raw_ob = raw_obs[agent]
            puf_ob = puf_obs[agent]

            # Reconstruct original obs format from puffer env and compare to raw
            if puf_env.is_obs_emulated:
                puf_ob = pufferlib.emulation.nativize(puf_ob, puf_env.env.single_observation_space, puf_env.obs_dtype)

            assert pufferlib.utils.compare_space_samples(raw_ob, puf_ob)

        raw_actions = {a: raw_env.action_space(a).sample() for a in raw_env.agents}
        puf_actions = raw_actions

        raw_obs, raw_rewards, raw_dones, raw_truncateds, _ = raw_env.step(raw_actions)

        # Convert raw actions to puffer format
        dummy_action = raw_actions[list(raw_actions.keys())[0]]
        if puf_env.is_atn_emulated:
            for agent in puf_env.possible_agents:
                if agent not in raw_actions:
                    puf_actions[agent] = dummy_action
                    continue

                puf_actions[agent] = pufferlib.emulation.emulate_copy(
                    raw_actions[agent], puf_env.single_action_space.dtype, puf_env.atn_dtype
                )

        puf_obs, puf_rewards, puf_dones, puf_truncateds, _ = puf_env.step(puf_actions)

        for agent in raw_rewards:
            assert puf_rewards[agent] == raw_rewards[agent]

        for agent in raw_dones:
            assert puf_dones[agent] == raw_dones[agent]


def test_puffer_vectorization(env_cls, puffer_cls, steps=100, num_envs=1, **kwargs):
    raw_envs = [puffer_cls(env_creator=env_cls) for _ in range(num_envs)]
    vec_envs = pufferlib.vector.make(puffer_cls, env_kwargs={"env_creator": env_cls}, num_envs=num_envs, **kwargs)

    num_agents = sum(env.num_agents for env in raw_envs)
    assert num_agents == vec_envs.num_agents

    raw_obs = [env.reset()[0] for i, env in enumerate(raw_envs)]
    vec_obs, _ = vec_envs.reset()

    for _ in range(steps):
        # PettingZoo dict observations
        if isinstance(raw_obs[0], dict):
            raw_obs = [v for d in raw_obs for v in d.values()]

        raw_obs = np.stack(raw_obs, axis=0)
        assert raw_obs.shape == vec_obs.shape
        assert np.all(raw_obs == vec_obs)

        actions = vec_envs.action_space.sample()
        raw_actions = np.split(actions, num_envs)

        # Copy reset behavior of VecEnv
        raw_obs, raw_rewards, raw_terminals, raw_truncations = [], [], [], []
        for idx, r_env in enumerate(raw_envs):
            if r_env.done:
                raw_obs.append(r_env.reset()[0])
                raw_rewards.extend([0] * r_env.num_agents)
                raw_terminals.extend([False] * r_env.num_agents)
                raw_truncations.extend([False] * r_env.num_agents)
            else:
                r_ob, r_rew, r_term, r_trunc, _ = r_env.step(raw_actions[idx])
                raw_obs.append(r_ob)
                raw_rewards.append(r_rew)
                raw_terminals.append(r_term)
                raw_truncations.append(r_trunc)

        vec_obs, vec_rewards, vec_terminals, vec_truncations, _ = vec_envs.step(actions)

        rew = raw_rewards
        if isinstance(raw_rewards[0], dict):
            raw_rewards = [v for d in raw_rewards for v in d.values()]
            raw_terminals = [v for d in raw_terminals for v in d.values()]
            raw_truncations = [v for d in raw_truncations for v in d.values()]

        raw_rewards = np.asarray(raw_rewards, dtype=np.float32)
        raw_terminals = np.asarray(raw_terminals)
        raw_truncations = np.asarray(raw_truncations)

        assert np.all(raw_rewards == vec_rewards)
        assert np.all(raw_terminals == vec_terminals)
        assert np.all(raw_truncations == vec_truncations)

    vec_envs.close()
    for raw_env in raw_envs:
        raw_env.close()


def test_emulation():
    for env_cls in test.MOCK_SINGLE_AGENT_ENVIRONMENTS:
        test_gymnasium_emulation(env_cls)

    print("Gymnasium emulation tests passed")

    for env_cls in test.MOCK_MULTI_AGENT_ENVIRONMENTS:
        test_pettingzoo_emulation(env_cls)

    print("PettingZoo emulation tests passed")


def test_vectorization():
    for vectorization in [pufferlib.vector.Serial, pufferlib.vector.Multiprocessing, pufferlib.vector.Ray]:
        for env_cls in test.MOCK_SINGLE_AGENT_ENVIRONMENTS:
            test_puffer_vectorization(
                env_cls,
                pufferlib.emulation.GymnasiumPufferEnv,
                steps=10,
                num_envs=4,
                num_workers=4,
                backend=vectorization,
            )

        print(f"Gymnasium {vectorization.__name__} vectorization tests passed")

        for env_cls in test.MOCK_MULTI_AGENT_ENVIRONMENTS:
            test_puffer_vectorization(
                env_cls,
                pufferlib.emulation.PettingZooPufferEnv,
                steps=10,
                num_envs=4,
                num_workers=4,
                backend=vectorization,
            )

        print(f"PettingZoo {vectorization.__name__} vectorization tests passed")


if __name__ == "__main__":
    test_emulation()
    test_vectorization()
    exit(0)  # For Ray
