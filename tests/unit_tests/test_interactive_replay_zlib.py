import json
import struct
import zlib

import numpy as np

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


def test_standard_replay_zlib_round_trip_and_html_render(tmp_path, monkeypatch):
    scenario = {
        "map_name": "test_map.bin",
        "scenario_id": "scenario_1",
        "active_agent_indices": [0],
        "agents": [
            {"mark_as_expert": 0},
            {"mark_as_expert": 1},
            {"mark_as_expert": 0},
        ],
        "road_elements": [],
        "traffic_elements": [],
    }
    replay_path = tmp_path / "episode.replay.zlib"
    html_path = tmp_path / "episode.html"

    compressed_payload = viz.save_interactive_replay_zlib(scenario, _standard_replay(), replay_path)
    monkeypatch.setattr(viz, "PAYLOAD_CHUNK_SIZE", 64)
    viz.render_interactive_replay_zlib(replay_path, html_path)

    assert replay_path.read_bytes() == compressed_payload
    html = html_path.read_text()
    assert "__PAYLOAD_CHUNKS__" not in html
    assert html.count('class="payload-chunk"') > 1
    assert "decodeReplayPayload()" in html
    assert 'const DYNAMIC_EXPERT_COLOR = "#c4c8cf";' in html
    assert 'const STATIC_AGENT_COLOR = "#4a505a";' in html
    assert 'const INFRACTION_AGENT_COLOR = "#d92d20";' in html
    assert "function agentHasInfraction(frame, idx)" in html
    assert "function colorForAgent(id, isActive, isExpert, hasInfraction)" in html
    assert "const hasInfraction = agentType === 1 && agentHasInfraction(frame, idx);" in html
    assert "function colorFor(id, isActive, isExpert)" in html
    assert "if (isActive) return VEHICLE_COLORS" in html
    assert "return isExpert ? DYNAMIC_EXPERT_COLOR : STATIC_AGENT_COLOR;" in html
    assert 'stopped ? "red"' not in html
    assert "inferLegacyDynamicAgents" not in html
    assert "road_width_to_position" not in html
    assert 'id="meta-obs-road"' not in html

    payload = zlib.decompress(compressed_payload)
    header_length = struct.unpack_from("<I", payload)[0]
    header = json.loads(payload[4 : 4 + header_length])
    assert header["expert_indices"] == [1]
    for unused_metadata_key in (
        "goal_regen_mode",
        "active_indices",
        "has_obs",
        "obs_slots_lane_n",
        "obs_slots_boundary_n",
        "obs_dropout_lane",
        "obs_dropout_boundary",
        "obs_lane_stride",
        "obs_boundary_stride",
    ):
        assert unused_metadata_key not in header
