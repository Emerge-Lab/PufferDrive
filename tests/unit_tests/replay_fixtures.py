"""Synthetic interactive replays and directory listings shared by the replay, render and tidy tests."""

import numpy as np

from opengigaflow import viz


AGENT_F32_FIELDS = 13
AGENT_I32_FIELDS = 10


def blank_replay(frame_count, agent_cap):
    """All-zero frames with unit actions, in the dict layout save_interactive_replay_zlib takes."""
    return {
        "env": {
            "num_goals": 2,
            "goal_radius": 2.0,
            "reward_conditioning": False,
            "obs_slots_partners_n": 4,
            "obs_slots_lane_n": 4,
            "obs_slots_boundary_n": 4,
            "obs_slots_traffic_controls_n": 1,
        },
        "agent_f32": np.zeros((frame_count, agent_cap, AGENT_F32_FIELDS), np.float32),
        "agent_i32": np.zeros((frame_count, agent_cap, AGENT_I32_FIELDS), np.int32),
        "metrics_f32": np.zeros((frame_count, agent_cap, len(viz.METRIC_LABELS)), np.float32),
        "traffic_i16": np.zeros((frame_count, 1, 3), np.int16),
        "raw_action": np.ones((frame_count, 1, 2), np.float32),
        "clipped_action": np.ones((frame_count, 1, 2), np.float32),
        "value": np.zeros((frame_count, 1), np.float32),
        "entropy": np.zeros((frame_count, 1), np.float32),
        "obs": None,
    }


def relative_tree(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))
