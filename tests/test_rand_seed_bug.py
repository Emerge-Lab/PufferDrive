"""Test that vec_reset produces diverse rand() seeds across workers.

Regression test for https://github.com/Emerge-Lab/PufferDrive/pull/355

Bug: vec_reset called srand(i + seed * num_envs) which ignores PID,
so all forked workers got identical rand() sequences. This caused
correlated reward coefficients, goal positions, and spawn positions.
"""

from multiprocessing import Pipe, Process

import numpy as np


def _worker_reward_coefs(conn, seed):
    """Forked child: create a Drive env, reset, and send back reward coefs."""
    from pufferlib.ocean.drive.drive import Drive

    env = Drive(
        num_agents=1,
        map_dir="resources/drive/binaries/carla_data",
        num_maps=1,
        init_mode="init_variable_agent_number",
        min_agents_per_env=1,
        max_agents_per_env=1,
        episode_length=300,
        seed=seed,
        reward_randomization=1,
        reward_conditioning=1,
    )
    env.reset()
    obs, _, _, _, _ = env.step(np.zeros((1, 1), dtype=np.int32))
    ego_dim = env.ego_features
    # TODO: use reward_coefs_from_obs() or similar instead of hardcoded magic index
    reward_section = obs[0, ego_dim - 16 : ego_dim]
    conn.send(tuple(round(float(x), 6) for x in reward_section))
    env.close()
    conn.close()


def test_forked_workers_get_different_reward_coefs():
    """Forked workers with the same seed must produce different reward coefs.

    Before the fix, srand(i + seed * num_envs) ignored PID, so all forked
    workers got identical rand() sequences and thus identical reward
    coefficients, goal positions, and spawn positions.
    """
    num_workers = 8
    seed = 42

    # Use bare multiprocessing.Process to match PufferLib's vectorization
    # (pufferlib/vector.py). On Linux this defaults to fork, which is where
    # the bug manifests. On macOS it defaults to spawn, which is also fine.
    coefs = []
    for _ in range(num_workers):
        parent_conn, child_conn = Pipe()
        p = Process(target=_worker_reward_coefs, args=(child_conn, seed))
        p.start()
        coefs.append(parent_conn.recv())
        p.join()

    unique = len(set(coefs))
    assert unique == num_workers, (
        f"Expected {num_workers} unique reward coefficient sets across forked "
        f"workers, but got {unique}. Workers are getting correlated random state."
    )


def test_same_process_resets_get_different_reward_coefs():
    """Repeated env creation + reset in the same process must vary."""
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
        # TODO: use reward_coefs_from_obs() or similar instead of hardcoded magic index
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
    test_forked_workers_get_different_reward_coefs()
    print("PASS: forked workers get different reward coefs")
    test_same_process_resets_get_different_reward_coefs()
    print("PASS: same-process resets get different reward coefs")
    print("All tests passed!")
