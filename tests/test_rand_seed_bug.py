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
        map_dir="resources/drive/binaries/carla_3D",
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
            map_dir="resources/drive/binaries/carla_3D",
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


def test_sub_envs_get_different_spawn_positions():
    """Sub-envs within a single Drive instance must spawn agents at different positions.

    Before the fix, all sub-envs got srand(seed) with the same seed in env_init,
    so spawn_agent produced identical positions for every sub-env.
    """
    from pufferlib.ocean.drive.drive import Drive

    env = Drive(
        num_agents=16,
        map_dir="resources/drive/binaries/carla_3D",
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
    obs, _, _, _, _ = env.step(np.zeros((env.num_agents, 1), dtype=np.int32))

    # Extract ego positions (obs indices 0-2 are relative goal, 3 is speed, etc.)
    # The actual sim positions aren't in the obs, but if spawn positions are the same,
    # the observations will be identical (same road context, same relative geometry).
    # Check that observations differ across agents.
    obs_tuples = [tuple(round(float(x), 4) for x in obs[i, :16]) for i in range(env.num_agents)]
    unique = len(set(obs_tuples))
    assert unique > env.num_agents // 2, (
        f"Only {unique}/{env.num_agents} unique ego observations across sub-envs. "
        f"Sub-envs likely have identical spawn positions (env_init seed not diversified)."
    )
    env.close()


def test_cross_worker_spawn_positions_differ():
    """Sub-envs across different workers (consecutive seeds) must not overlap.

    Workers get seed, seed+1, seed+2, etc. from pufferlib. Sub-envs within
    each worker get seed*num_envs+i. Without the stride, worker 0's sub-env 1
    and worker 1's sub-env 0 would both get srand(seed+1).
    """
    from pufferlib.ocean.drive.drive import Drive

    all_obs = []
    for worker_seed in [42, 43]:  # two consecutive workers
        env = Drive(
            num_agents=4,
            map_dir="resources/drive/binaries/carla_3D",
            num_maps=1,
            init_mode="init_variable_agent_number",
            min_agents_per_env=1,
            max_agents_per_env=1,
            episode_length=300,
            seed=worker_seed,
            reward_randomization=1,
            reward_conditioning=1,
        )
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros((env.num_agents, 1), dtype=np.int32))
        for i in range(env.num_agents):
            all_obs.append(tuple(round(float(x), 4) for x in obs[i, :16]))
        env.close()

    unique = len(set(all_obs))
    total = len(all_obs)
    assert unique == total, (
        f"Only {unique}/{total} unique ego observations across 2 workers. Cross-worker sub-envs have overlapping seeds."
    )


if __name__ == "__main__":
    test_forked_workers_get_different_reward_coefs()
    print("PASS: forked workers get different reward coefs")
    test_same_process_resets_get_different_reward_coefs()
    print("PASS: same-process resets get different reward coefs")
    test_sub_envs_get_different_spawn_positions()
    print("PASS: sub-envs get different spawn positions")
    test_cross_worker_spawn_positions_differ()
    print("PASS: cross-worker spawn positions differ")
    print("All tests passed!")
