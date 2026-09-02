"""Explicit ghost (logged/human trajectory) in interactive replays: encoder override + in-place rewrite."""

from types import SimpleNamespace

import numpy as np
import pytest

from pufferlib import viz
from pufferlib.ocean.cosim import nuplan_bridge as nb

FRAMES, AGENT_CAP, TRAFFIC_CAP = 6, 3, 2


def _replay(ghost=None):
    replay = {
        "env": {
            "num_goals": 2,
            "reward_conditioning": False,
            "obs_slots_partners_n": 4,
            "obs_slots_lane_n": 4,
            "obs_slots_boundary_n": 4,
            "obs_slots_traffic_controls_n": 1,
        },
        "agent_f32": np.zeros((FRAMES, AGENT_CAP, 12), np.float32),
        "agent_i32": np.zeros((FRAMES, AGENT_CAP, 10), np.int32),
        "metrics_f32": np.zeros((FRAMES, AGENT_CAP, 4), np.float32),
        "puffer_f32": np.zeros((FRAMES, AGENT_CAP, 4), np.float32),
        "traffic_i16": np.zeros((FRAMES, TRAFFIC_CAP, 3), np.int16),
        "raw_action": np.ones((FRAMES, 1, 2), np.float32),
        "clipped_action": np.ones((FRAMES, 1, 2), np.float32),
        "value": np.zeros((FRAMES, 1), np.float32),
        "entropy": np.zeros((FRAMES, 1), np.float32),
        "obs": None,
    }
    if ghost is not None:
        replay["ghost_f32"] = ghost
    return replay


def _scenario_with_logged_agent():
    return {
        "num_total_agents": AGENT_CAP,
        "active_agent_indices": [0],
        "agents": [
            {
                "log_trajectory_x": list(range(FRAMES)),
                "log_trajectory_y": [0.0] * FRAMES,
                "log_heading": [0.0] * FRAMES,
                "log_valid": [1] * FRAMES,
                "sim_length": 4.0,
                "sim_width": 2.0,
            }
        ],
    }


def _ghost_values():
    ghost = np.zeros((FRAMES, 1, 5), np.float32)
    ghost[:, 0, 0] = np.arange(FRAMES) * 10.0
    ghost[:, 0, 1] = 7.0
    ghost[:, 0, 2] = 0.5
    ghost[:, 0, 3] = 5.0
    ghost[:, 0, 4] = 2.3
    return ghost


def test_explicit_ghost_overrides_scenario_log(tmp_path):
    path = tmp_path / "a.replay.zlib"
    ghost = _ghost_values()
    viz.save_interactive_replay_zlib(_scenario_with_logged_agent(), _replay(ghost), str(path))
    _, chunks = viz.read_replay_zlib(path)
    assert np.array_equal(chunks["ghost_f32"], ghost)


def test_scenario_log_still_used_without_explicit_ghost(tmp_path):
    path = tmp_path / "b.replay.zlib"
    viz.save_interactive_replay_zlib(_scenario_with_logged_agent(), _replay(), str(path))
    _, chunks = viz.read_replay_zlib(path)
    assert np.array_equal(chunks["ghost_f32"][:, 0, 0], np.arange(FRAMES, dtype=np.float32))
    assert np.all(chunks["ghost_f32"][:, 0, 4] == 2.0)


def test_explicit_ghost_shape_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="ghost_f32 shape"):
        viz.save_interactive_replay_zlib({}, _replay(_ghost_values()[:2]), str(tmp_path / "c.replay.zlib"))


def test_set_replay_ghost_rewrites_only_the_ghost_chunk(tmp_path):
    path = tmp_path / "d.replay.zlib"
    viz.save_interactive_replay_zlib({}, _replay(), str(path))
    header_before, chunks_before = viz.read_replay_zlib(path)
    chunks_before = {name: arr.copy() for name, arr in chunks_before.items()}
    assert not chunks_before["ghost_f32"].any()

    ghost = _ghost_values()
    viz.set_replay_ghost(path, ghost)
    header_after, chunks_after = viz.read_replay_zlib(path)
    assert header_after == header_before
    for name, arr in chunks_before.items():
        expected = ghost if name == "ghost_f32" else arr
        assert np.array_equal(chunks_after[name], expected), name
    with pytest.raises(ValueError, match="ghost chunk shape"):
        viz.set_replay_ghost(path, ghost[:1])


def test_logged_ego_boxes_in_bin_frame():
    def ego_state(i):
        return SimpleNamespace(
            center=SimpleNamespace(x=100.0 + i, y=200.0, heading=0.25),
            car_footprint=SimpleNamespace(length=5.1, width=2.3),
        )

    scenario = SimpleNamespace(get_number_of_iterations=lambda: 4, get_ego_state_at_iteration=ego_state)
    boxes = nb.logged_ego_boxes(scenario, nb.NuPlanTransform(100.0, 200.0))
    assert boxes.shape == (4, 5) and boxes.dtype == np.float32
    assert np.allclose(boxes[:, 0], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(boxes[:, 1], 0.0)
    assert np.allclose(boxes[:, 2:], [0.25, 5.1, 2.3])
