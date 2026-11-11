"""obs_lane_width_clip_{min,max}_m clip only the lane-row width column of the observation."""

import os
import shutil
import sys

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "data_utils"))
from mirror_map_bin import _is_lane, read_bin, write_bin  # noqa: E402

SOURCE_BIN = os.path.join(
    REPO_ROOT,
    "pufferlib",
    "resources",
    "drive",
    "binaries",
    "sdc_replay_test",
    "nuplan__c2a99be0-5b3e-5e67-88e6-6fd0f22e5419.bin",
)
LANE_WIDTH_COLUMN = 6
CLIP_MIN_M = 2.85
CLIP_MAX_M = 4.4
ROAD_SEG_WIDTH_NORM_M = 5.0


def _write_uniform_width_bin(out_dir, lane_width_m):
    data = read_bin(SOURCE_BIN)
    for road in data["roads"]:
        if _is_lane(road["type"]):
            road["widths"] = tuple([float(lane_width_m)] * road["S"])
    data["has_phase_section"] = True  # the loader reads the tail sections in order: phases, widths, zones
    data["has_width_section"] = True
    os.makedirs(out_dir)
    write_bin(data, os.path.join(out_dir, os.path.basename(SOURCE_BIN)))
    return out_dir


def _lane_width_obs(map_dir, **clip_kwargs):
    env = Drive(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=map_dir,
        simulation_mode="replay",
        goal_source="gt",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=400,
        resample_frequency=1_000_000,
        termination_mode=False,
        terminate_on_goal=False,
        obs_dropout_lane=0.0,
        obs_dropout_boundary=0.0,
        obs_norm_road_seg_width_m=ROAD_SEG_WIDTH_NORM_M,
        **clip_kwargs,
    )
    obs, _ = env.reset()
    lane_start = (
        env.ego_features + env.num_reward_coefs + env.goal_dim + env.obs_slots_partners_n * env.partner_features
    )
    lane_rows = obs[:, lane_start : lane_start + env.obs_slots_lane_kept * env.lane_features]
    lane_rows = lane_rows.reshape(len(obs), env.obs_slots_lane_kept, env.lane_features)
    lane_counts = obs[:, -4].astype(int)
    env.close()
    widths = [lane_rows[agent, :count, LANE_WIDTH_COLUMN] for agent, count in enumerate(lane_counts)]
    widths = np.concatenate(widths)
    assert len(widths) > 0
    return widths


@pytest.mark.parametrize("lane_width_m,expected_clipped_m", [(12.0, CLIP_MAX_M), (1.0, CLIP_MIN_M)])
def test_lane_width_clipped_to_range(tmp_path, lane_width_m, expected_clipped_m):
    map_dir = _write_uniform_width_bin(str(tmp_path / "bins"), lane_width_m)
    unclipped = _lane_width_obs(map_dir)
    clipped = _lane_width_obs(map_dir, obs_lane_width_clip_min_m=CLIP_MIN_M, obs_lane_width_clip_max_m=CLIP_MAX_M)
    np.testing.assert_allclose(unclipped, lane_width_m / ROAD_SEG_WIDTH_NORM_M, rtol=1e-5)
    np.testing.assert_allclose(clipped, expected_clipped_m / ROAD_SEG_WIDTH_NORM_M, rtol=1e-5)
    shutil.rmtree(map_dir)


def test_lane_width_clip_disabled_by_default_and_by_zero_max(tmp_path):
    map_dir = _write_uniform_width_bin(str(tmp_path / "bins"), 12.0)
    default = _lane_width_obs(map_dir)
    zero_max = _lane_width_obs(map_dir, obs_lane_width_clip_min_m=CLIP_MIN_M, obs_lane_width_clip_max_m=0.0)
    np.testing.assert_allclose(default, 12.0 / ROAD_SEG_WIDTH_NORM_M, rtol=1e-5)
    np.testing.assert_array_equal(zero_max, default)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(obs_lane_width_clip_min_m=-1.0),
        dict(obs_lane_width_clip_max_m=float("nan")),
        dict(obs_lane_width_clip_min_m=5.0, obs_lane_width_clip_max_m=4.4),
    ],
)
def test_lane_width_clip_validation(kwargs):
    with pytest.raises(ValueError, match="obs_lane_width_clip"):
        Drive(config_only=True, **kwargs)
