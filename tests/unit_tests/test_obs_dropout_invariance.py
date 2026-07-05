"""Road-observation dropout must not change the observation interface.

Dropout is env noise: it limits how many lane/boundary slots carry data each
step, but the obs vector is always laid out for the full slot counts, with
unused slots zeroed and the per-agent valid-count features reporting what was
kept. Two envs differing only in dropout therefore share an observation space,
so a policy trained with dropout can be evaluated in a clean env (the
EvalManager clean macro zeroes dropout) and vice versa.

Fixture map: the single checked-in nuPlan replay .bin under
pufferlib/resources/drive/binaries/sdc_replay_test/.
"""

import os

import numpy as np

from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")

LANE_SLOTS = 20
BOUNDARY_SLOTS = 10
ROAD_FEATURES = 7


def _make_env(**overrides):
    kwargs = dict(
        num_agents=4,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=80,
        resample_frequency=1_000_000,
        termination_mode=0,
        obs_slots_lane_n=LANE_SLOTS,
        obs_slots_boundary_n=BOUNDARY_SLOTS,
    )
    kwargs.update(overrides)
    return Drive(**kwargs)


def test_obs_size_invariant_to_dropout():
    clean = _make_env()
    noisy = _make_env(obs_dropout_lane=0.5, obs_dropout_boundary=0.3)
    assert clean.num_obs == noisy.num_obs
    assert clean.single_observation_space.shape == noisy.single_observation_space.shape
    clean.close()
    noisy.close()


def test_dropout_limits_valid_counts_and_zeroes_padding():
    env = _make_env(obs_dropout_lane=0.5, obs_dropout_boundary=0.3)
    lane_kept = env.obs_slots_lane_kept
    boundary_kept = env.obs_slots_boundary_kept
    assert lane_kept == int(LANE_SLOTS * 0.5)
    assert boundary_kept == int(BOUNDARY_SLOTS * 0.7)

    obs, _ = env.reset(seed=0)
    # Obs write order: ego, reward+target, partners, lanes, boundaries, ...
    lane_start = (
        env.ego_features
        + env.num_reward_coefs
        + env.target_dim
        + env.obs_slots_partners_n * env.partner_features
    )
    boundary_start = lane_start + LANE_SLOTS * ROAD_FEATURES
    # Valid-count features are the last block of the vector
    counts_start = env.num_obs - env.obs_valid_count_features

    for row in np.asarray(obs):
        lane_count = int(row[counts_start])
        boundary_count = int(row[counts_start + 1])
        assert 0 <= lane_count <= lane_kept
        assert 0 <= boundary_count <= boundary_kept
        lane_block = row[lane_start : lane_start + LANE_SLOTS * ROAD_FEATURES]
        boundary_block = row[boundary_start : boundary_start + BOUNDARY_SLOTS * ROAD_FEATURES]
        # Slots beyond the kept count are pure padding
        assert not lane_block[lane_count * ROAD_FEATURES :].any()
        assert not boundary_block[boundary_count * ROAD_FEATURES :].any()
    env.close()


def test_policy_transfers_between_dropout_settings():
    """The exact nightly regression: a policy built against a dropout env must
    run on observations from a clean env (and the reverse)."""
    import torch

    from pufferlib.ocean.torch import Policy

    noisy = _make_env(obs_dropout_lane=0.5, obs_dropout_boundary=0.3)
    clean = _make_env()
    policy = Policy(noisy)

    for env in (clean, noisy):
        obs, _ = env.reset(seed=0)
        with torch.no_grad():
            logits, values = policy(torch.as_tensor(np.asarray(obs)))
        assert values.shape[0] == obs.shape[0]

    clean.close()
    noisy.close()
