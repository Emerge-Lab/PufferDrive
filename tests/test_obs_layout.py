"""Test that ObsLayout matches what compute_observations actually writes.

If someone changes the observation layout in C without updating
compute_obs_layout(), these tests will fail.
"""

import numpy as np
import pytest


def make_env():
    from pufferlib.ocean.drive.drive import Drive
    return Drive(
        num_agents=4,
        map_dir="resources/drive/binaries/carla_3D",
        num_maps=1,
        init_mode="init_variable_agent_number",
        min_agents_per_env=1,
        max_agents_per_env=4,
        episode_length=300,
        seed=42,
        reward_randomization=1,
        reward_conditioning=1,
        dynamics_model="jerk",
    )


def test_obs_total_matches_actual_obs_shape():
    """obs_layout.num_obs must match the actual observation array width."""
    env = make_env()
    env.reset()
    obs, _, _, _, _ = env.step(np.zeros((env.num_agents, 1), dtype=np.int32))
    layout = env.obs_layout
    assert obs.shape[-1] == layout.num_obs, (
        f"obs width {obs.shape[-1]} != layout.num_obs {layout.num_obs}"
    )
    env.close()


def test_ego_section_nonzero():
    """Ego features should be nonzero after a step (speed, goal dist, etc.)."""
    env = make_env()
    env.reset()
    obs, _, _, _, _ = env.step(np.zeros((env.num_agents, 1), dtype=np.int32))
    layout = env.obs_layout
    ego = layout.ego(obs[0])
    assert ego.shape[-1] == layout.ego_dim
    # At least some ego features should be nonzero
    assert np.any(ego != 0), "Ego features are all zero — layout may be wrong"
    env.close()


def test_reward_coefs_in_expected_range():
    """Reward coefs should be nonzero when reward_randomization=1."""
    env = make_env()
    env.reset()
    obs, _, _, _, _ = env.step(np.zeros((env.num_agents, 1), dtype=np.int32))
    layout = env.obs_layout
    coefs = layout.reward_coefs(obs[0])
    assert coefs is not None, "reward_coefs returned None with conditioning on"
    assert coefs.shape[-1] == layout.num_reward_coefs
    assert np.any(coefs != 0), (
        "Reward coefs are all zero — obs_reward_coef_start may be wrong"
    )
    env.close()


def test_partner_section_shape():
    """Partner features should reshape to (max_partners, partner_features)."""
    env = make_env()
    env.reset()
    obs, _, _, _, _ = env.step(np.zeros((env.num_agents, 1), dtype=np.int32))
    layout = env.obs_layout
    partners = layout.partners(obs[0])
    assert partners.shape == (layout.max_partner_objects, layout.partner_features)
    env.close()


def test_road_section_shape():
    """Road features should reshape to (max_roads, road_features)."""
    env = make_env()
    env.reset()
    obs, _, _, _, _ = env.step(np.zeros((env.num_agents, 1), dtype=np.int32))
    layout = env.obs_layout
    roads = layout.roads(obs[0])
    assert roads.shape == (layout.max_road_objects, layout.road_features)
    env.close()


def test_sections_are_contiguous_and_non_overlapping():
    """Ego, partner, and road sections should tile the full obs without gaps."""
    env = make_env()
    layout = env.obs_layout
    assert layout.ego_dim == layout._partner_start, (
        f"Gap between ego ({layout.ego_dim}) and partners ({layout._partner_start})"
    )
    expected_partner_end = layout._partner_start + layout.max_partner_objects * layout.partner_features
    assert expected_partner_end == layout._road_start, (
        f"Gap between partners end ({expected_partner_end}) and roads ({layout._road_start})"
    )
    expected_road_end = layout._road_start + layout.max_road_objects * layout.road_features
    assert expected_road_end == layout.num_obs, (
        f"Road end ({expected_road_end}) != num_obs ({layout.num_obs})"
    )
    env.close()


def test_road_categorical_feature_valid():
    """The last road feature is a categorical type (0-6). Verify it's in range."""
    env = make_env()
    env.reset()
    obs, _, _, _, _ = env.step(np.zeros((env.num_agents, 1), dtype=np.int32))
    layout = env.obs_layout
    roads = layout.roads(obs[0])
    # Find non-zero road entries (zero means empty/padding)
    nonzero_mask = np.any(roads[:, :3] != 0, axis=1)
    if np.any(nonzero_mask):
        categorical = roads[nonzero_mask, -1]
        assert np.all(categorical >= 0) and np.all(categorical <= 6), (
            f"Road categorical feature out of range [0,6]: {categorical}"
        )
    env.close()


if __name__ == "__main__":
    test_obs_total_matches_actual_obs_shape()
    test_ego_section_nonzero()
    test_reward_coefs_in_expected_range()
    test_partner_section_shape()
    test_road_section_shape()
    test_sections_are_contiguous_and_non_overlapping()
    test_road_categorical_feature_valid()
    print("All obs layout tests passed!")
