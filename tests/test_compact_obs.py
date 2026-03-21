"""Test that compact obs reconstruction produces identical road features to full obs mode."""

import numpy as np
import pytest


def make_env(compact_obs, num_agents=8, seed=42):
    from pufferlib.ocean.drive.drive import Drive

    return Drive(
        num_agents=num_agents,
        map_dir="resources/drive/binaries/carla_data",
        num_maps=1,
        init_mode="init_variable_agent_number",
        min_agents_per_env=1,
        max_agents_per_env=1,
        episode_length=300,
        compact_obs=compact_obs,
        seed=seed,
    )


def test_compact_obs_reconstruction_matches_full():
    """Verify that reconstructed road obs from compact mode exactly match full obs mode."""
    from pufferlib.ocean.drive import binding

    # Create both envs with same seed so agents spawn at same positions
    env_full = make_env(compact_obs=False, seed=42)
    env_compact = make_env(compact_obs=True, seed=42)

    assert env_full.num_obs == 3080
    assert env_compact.num_obs == env_compact.ego_partner_dim + 5

    # Reset both
    env_full.reset()
    env_compact.reset()

    # Step both with same actions
    actions_full = np.zeros((env_full.num_agents, 1), dtype=np.int32)
    actions_compact = np.zeros((env_compact.num_agents, 1), dtype=np.int32)

    full_obs, _, _, _, _ = env_full.step(actions_full)
    compact_obs, _, _, _, _ = env_compact.step(actions_compact)

    # Reconstruct from compact
    n = compact_obs.shape[0]
    reconstructed = np.zeros((n, env_compact.full_num_obs), dtype=np.float32)
    binding.reconstruct_road_obs(
        np.ascontiguousarray(compact_obs.astype(np.float32)),
        reconstructed,
        env_compact.ego_partner_dim,
        env_compact.full_road_dim,
    )

    # Compare ego + partner features (should be identical)
    ego_partner_dim = env_compact.ego_partner_dim
    np.testing.assert_allclose(
        reconstructed[:, :ego_partner_dim],
        full_obs[:, :ego_partner_dim],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Ego + partner features differ between compact and full obs",
    )

    # Compare road features (should be identical since same position + same map)
    road_full = full_obs[:, ego_partner_dim:]
    road_reconstructed = reconstructed[:, ego_partner_dim:]

    np.testing.assert_allclose(
        road_reconstructed,
        road_full,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Reconstructed road features differ from full obs road features",
    )

    print(f"PASS: {n} agents, road features match exactly")
    print(f"  Full obs shape: {full_obs.shape}")
    print(f"  Compact obs shape: {compact_obs.shape}")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Road features nonzero: {np.count_nonzero(road_full)} / {road_full.size}")

    env_full.close()
    env_compact.close()


def test_compact_obs_multiple_steps():
    """Verify reconstruction matches over multiple steps as agents move."""
    from pufferlib.ocean.drive import binding

    env_full = make_env(compact_obs=False, seed=123)
    env_compact = make_env(compact_obs=True, seed=123)
    env_full.reset()
    env_compact.reset()

    for step in range(10):
        actions = np.zeros((env_full.num_agents, 1), dtype=np.int32)
        full_obs, _, _, _, _ = env_full.step(actions)
        compact_obs, _, _, _, _ = env_compact.step(actions)

        n = compact_obs.shape[0]
        reconstructed = np.zeros((n, env_compact.full_num_obs), dtype=np.float32)
        binding.reconstruct_road_obs(
            np.ascontiguousarray(compact_obs.astype(np.float32)),
            reconstructed,
            env_compact.ego_partner_dim,
            env_compact.full_road_dim,
        )

        np.testing.assert_allclose(
            reconstructed,
            full_obs,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Mismatch at step {step}",
        )

    print(f"PASS: 10 steps, all road features match")

    env_full.close()
    env_compact.close()


def test_compact_obs_shape():
    """Verify obs shapes are correct in both modes."""
    env_compact = make_env(compact_obs=True)
    env_full = make_env(compact_obs=False)

    assert env_compact.num_obs == env_compact.ego_partner_dim + 5
    assert env_full.num_obs == env_full.ego_partner_dim + env_full.full_road_dim
    assert env_compact.full_num_obs == env_full.num_obs

    env_compact.close()
    env_full.close()
    print("PASS: obs shapes correct")


if __name__ == "__main__":
    test_compact_obs_shape()
    test_compact_obs_reconstruction_matches_full()
    test_compact_obs_multiple_steps()
    print("\nAll tests passed!")
