"""Test that vec_reset produces diverse rand() seeds across workers.

Regression test for https://github.com/Emerge-Lab/PufferDrive/pull/355

Bug: vec_reset called srand(i + seed * num_envs) which ignores PID,
so all forked workers got identical rand() sequences. This caused
correlated reward coefficients, goal positions, and spawn positions.
"""

import numpy as np
import ctypes


def test_vec_reset_srand_varies_across_pids():
    """After fork, different PIDs should produce different rand() sequences.
    The old code used srand(i + seed * num_envs) which is PID-independent.
    """
    libc = ctypes.CDLL(None)

    seed = 42
    num_envs = 1

    # Simulate old vec_reset: srand(i + seed * num_envs) — ignores PID
    old_seeds = []
    for fake_pid in range(16):
        srand_val = 0 + seed * num_envs
        libc.srand(srand_val)
        old_seeds.append(libc.rand())

    unique_old = len(set(old_seeds))
    assert unique_old > 1, (
        f"vec_reset srand produces identical rand() for all 16 simulated workers "
        f"({unique_old}/16 unique). srand(i + seed * num_envs) ignores PID — "
        f"all forked workers get correlated random state."
    )


def test_reward_coefs_vary_across_instances():
    """With reward_randomization=1, different env instances should generate
    different reward coefficients after reset.
    """
    from pufferlib.ocean.drive.drive import Drive

    reward_obs = []
    for _ in range(10):
        env = Drive(
            num_agents=1,
            map_dir="resources/drive/binaries/carla_data",
            num_maps=1,
            init_mode="init_variable_agent_number",
            min_agents_per_env=1,
            max_agents_per_env=1,
            episode_length=300,
            seed=42,
            reward_randomization=1,
            reward_conditioning=1,
        )
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros((1, 1), dtype=np.int32))
        ego_dim = env.ego_features
        reward_section = obs[0, ego_dim - 16 : ego_dim]
        reward_obs.append(tuple(round(float(x), 6) for x in reward_section))
        env.close()

    unique = len(set(reward_obs))
    assert unique > 5, (
        f"Only {unique}/10 unique reward coefficient sets across instances. "
        f"rand() seeding bug causes identical reward randomization across workers."
    )


if __name__ == "__main__":
    test_vec_reset_srand_varies_across_pids()
    test_reward_coefs_vary_across_instances()
    print("All tests passed!")
