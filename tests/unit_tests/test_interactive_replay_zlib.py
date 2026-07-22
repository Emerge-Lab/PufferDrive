import zlib

import numpy as np
import pytest

from pufferlib import viz
from pufferlib.ocean.drive import binding


def _standard_replay():
    frame_count = 2
    agent_count = 1
    return {
        "env": {
            "action_type": "discrete",
            "dynamics_model": "jerk",
            "goal_source": "route",
            "goal_regen_mode": "finite",
            "num_goals": 3,
            "reward_conditioning": False,
            "obs_slots_partners_n": 1,
            "obs_slots_lane_n": 1,
            "obs_slots_boundary_n": 1,
            "obs_slots_traffic_controls_n": 1,
            "obs_dropout_lane": 0.0,
            "obs_dropout_boundary": 0.0,
            "obs_lane_stride": 1,
            "obs_boundary_stride": 1,
            "obs_norm_goal_offset_m": 100.0,
            "obs_norm_xy_offset_m": 100.0,
            "obs_norm_veh_length_m": 15.0,
            "obs_norm_veh_width_m": 10.0,
            "obs_norm_road_seg_length_m": 5.0,
            "obs_norm_road_seg_width_m": 5.0,
        },
        "agent_f32": np.zeros((frame_count, agent_count, binding.AGENT_F32_FIELDS), dtype=np.float32),
        "agent_i32": np.zeros((frame_count, agent_count, binding.AGENT_I32_FIELDS), dtype=np.int32),
        "metrics_f32": np.zeros((frame_count, agent_count, binding.METRICS_F32_FIELDS), dtype=np.float32),
        "puffer_f32": np.zeros((frame_count, agent_count, binding.SCORE_F32_FIELDS), dtype=np.float32),
        "traffic_i16": np.zeros((frame_count, 1, binding.TRAFFIC_I16_FIELDS), dtype=np.int16),
        "obs": np.zeros((frame_count, agent_count, 8), dtype=np.float32),
        "raw_action": np.zeros((frame_count, agent_count, 1), dtype=np.float32),
        "clipped_action": np.zeros((frame_count, agent_count, 1), dtype=np.float32),
        "value": np.zeros((frame_count, agent_count), dtype=np.float32),
        "entropy": np.zeros((frame_count, agent_count), dtype=np.float32),
        "policy_probs": np.ones((frame_count, agent_count, 1), dtype=np.float32),
    }


def test_standard_replay_zlib_round_trip_and_html_render(tmp_path):
    scenario = {
        "map_name": "test_map.bin",
        "scenario_id": "scenario_1",
        "active_agent_indices": [0],
        "road_elements": [],
        "traffic_elements": [],
    }
    replay_path = tmp_path / "episode.replay.zlib"
    html_path = tmp_path / "episode.html"

    compressed_payload = viz.save_interactive_replay_zlib(scenario, _standard_replay(), replay_path)
    header = viz.validate_interactive_replay(compressed_payload)
    viz.render_interactive_replay_zlib(replay_path, html_path)

    assert header["schema"] == viz.REPLAY_SCHEMA
    assert header["frames"] == 2
    assert replay_path.read_bytes() == compressed_payload
    assert "__B64_PAYLOAD__" not in html_path.read_text()


def test_standard_replay_zlib_rejects_trailing_compressed_data():
    payload = viz.encode_interactive_replay(
        {"road_elements": [], "traffic_elements": [], "active_agent_indices": []}, _standard_replay()
    )

    with pytest.raises(ValueError, match="trailing"):
        viz.validate_interactive_replay(payload + zlib.compress(b"extra"))
