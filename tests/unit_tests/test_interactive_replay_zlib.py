"""Serialization contracts for portable interactive evaluation replays."""

import base64
import json
import re
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
        "agent_f32": np.arange(
            frame_count * agent_count * binding.AGENT_F32_FIELDS,
            dtype=np.float32,
        ).reshape(frame_count, agent_count, binding.AGENT_F32_FIELDS),
        "agent_i32": np.arange(
            frame_count * agent_count * binding.AGENT_I32_FIELDS,
            dtype=np.int32,
        ).reshape(frame_count, agent_count, binding.AGENT_I32_FIELDS),
        "metrics_f32": np.arange(
            frame_count * agent_count * binding.METRICS_F32_FIELDS,
            dtype=np.float32,
        ).reshape(frame_count, agent_count, binding.METRICS_F32_FIELDS),
        "puffer_f32": np.arange(
            frame_count * agent_count * binding.SCORE_F32_FIELDS,
            dtype=np.float32,
        ).reshape(frame_count, agent_count, binding.SCORE_F32_FIELDS),
        "traffic_i16": np.arange(
            frame_count * binding.TRAFFIC_I16_FIELDS,
            dtype=np.int16,
        ).reshape(frame_count, 1, binding.TRAFFIC_I16_FIELDS),
        "obs": np.asarray([[[-1.0, 0.0, 1.0]], [[1.0, 0.0, -1.0]]], dtype=np.float32),
        "raw_action": np.asarray([[[0.0]], [[1.0]]], dtype=np.float32),
        "clipped_action": np.asarray([[[0.0]], [[1.0]]], dtype=np.float32),
        "value": np.asarray([[0.25], [0.5]], dtype=np.float32),
        "entropy": np.asarray([[0.75], [0.5]], dtype=np.float32),
        "policy_probs": np.asarray([[[0.25, 0.75]], [[0.6, 0.4]]], dtype=np.float32),
    }


def _decode_packed_replay(compressed_payload):
    payload = zlib.decompress(compressed_payload)
    header_length = struct.unpack_from("<I", payload)[0]
    header_end = 4 + header_length
    header = json.loads(payload[4:header_end])
    data_start = header_end + (-header_end) % 4
    decoded_chunks = {}
    for chunk_name, chunk_metadata in header["chunks"].items():
        chunk_shape = tuple(chunk_metadata["shape"])
        chunk_count = int(np.prod(chunk_shape, dtype=np.int64))
        decoded_chunks[chunk_name] = np.frombuffer(
            payload,
            dtype=np.dtype(chunk_metadata["dtype"]),
            count=chunk_count,
            offset=data_start + chunk_metadata["offset"],
        ).reshape(chunk_shape)
    return header, decoded_chunks


def test_standard_replay_zlib_round_trip_and_html_render(tmp_path, monkeypatch):
    """A discrete replay survives binary packing and embedding in standalone HTML."""
    scenario = {
        "map_name": "test_map.bin",
        "scenario_id": "scenario_1",
        "num_total_agents": 1,
        "active_agent_indices": [0],
        "agents": [
            {
                "mark_as_expert": 1,
                "sim_length": 4.5,
                "sim_width": 2.0,
                "log_trajectory_x": [0.0, 1.0],
                "log_trajectory_y": [0.0, 0.5],
                "log_heading": [0.0, 0.05],
                "log_valid": [1, 1],
            },
        ],
        "road_elements": [],
        "traffic_elements": [],
    }
    replay_path = tmp_path / "episode.replay.zlib"
    html_path = tmp_path / "episode.html"
    replay = _standard_replay()

    compressed_payload = viz.save_interactive_replay_zlib(scenario, replay, replay_path)
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

    embedded_payload_chunks = re.findall(
        r'<script[^>]*class="payload-chunk"[^>]*>(.*?)</script>',
        html,
    )
    assert base64.b64decode("".join(embedded_payload_chunks)) == compressed_payload

    header, decoded_chunks = _decode_packed_replay(compressed_payload)
    assert header["expert_indices"] == [0]
    assert header["total_agents"] == 1
    assert header["agent_cap"] == 1
    assert header["active_count"] == 1
    assert header["obs_dim"] == replay["obs"].shape[2]
    for chunk_metadata in header["chunks"].values():
        assert chunk_metadata["offset"] % 4 == 0

    lossless_replay_chunks = (
        "agent_f32",
        "agent_i32",
        "metrics_f32",
        "puffer_f32",
        "traffic_i16",
        "raw_action",
        "clipped_action",
        "value",
        "entropy",
        "policy_probs",
    )
    for chunk_name in lossless_replay_chunks:
        np.testing.assert_array_equal(decoded_chunks[chunk_name], replay[chunk_name])
    np.testing.assert_allclose(
        decoded_chunks["obs"].astype(np.float32) * header["obs_scale"],
        replay["obs"],
        atol=header["obs_scale"] / 2 + np.finfo(np.float32).eps,
        rtol=0,
    )
    np.testing.assert_allclose(
        decoded_chunks["ghost_f32"][:, 0],
        [
            [0.0, 0.0, 0.0, 4.5, 2.0],
            [1.0, 0.5, 0.05, 4.5, 2.0],
        ],
    )
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


def test_binary_packer_round_trips_supported_dtypes():
    """Every dtype accepted by the replay schema retains its values and shape."""
    chunks = {
        "float_values": np.asarray([1.25, -2.5], dtype=np.float32),
        "int_values": np.asarray([1, -2], dtype=np.int32),
        "short_values": np.asarray([3, -4], dtype=np.int16),
        "byte_values": np.asarray([5, 255], dtype=np.uint8),
    }

    header, decoded_chunks = _decode_packed_replay(viz._pack_replay_binary({"schema": "test"}, chunks))

    assert header["schema"] == "test"
    for chunk_name, expected_values in chunks.items():
        np.testing.assert_array_equal(decoded_chunks[chunk_name], expected_values)


def test_continuous_policy_chunks_round_trip():
    """Continuous policies store distribution statistics instead of probabilities."""
    replay = _standard_replay()
    replay["env"]["action_type"] = "continuous"
    replay.pop("policy_probs")
    replay["raw_action"] = np.asarray([[[0.1, -0.2]], [[0.3, -0.4]]], dtype=np.float32)
    replay["clipped_action"] = np.asarray([[[0.1, -0.2]], [[0.25, -0.25]]], dtype=np.float32)
    replay["policy_mean"] = np.asarray([[[0.0, 0.1]], [[0.2, 0.3]]], dtype=np.float32)
    replay["policy_std"] = np.asarray([[[0.5, 0.6]], [[0.7, 0.8]]], dtype=np.float32)
    replay["policy_log_prob"] = np.asarray([[-0.25], [-0.5]], dtype=np.float32)
    scenario = {
        "active_agent_indices": [0],
        "agents": [{"mark_as_expert": 0}],
        "road_elements": [],
        "traffic_elements": [],
    }

    header, decoded_chunks = _decode_packed_replay(viz.encode_interactive_replay(scenario, replay))

    assert header["action_type"] == "continuous"
    assert "policy_probs" not in decoded_chunks
    for chunk_name in ("policy_mean", "policy_std", "policy_log_prob"):
        np.testing.assert_array_equal(decoded_chunks[chunk_name], replay[chunk_name])
