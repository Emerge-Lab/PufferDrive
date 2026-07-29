"""Smoke coverage from a real evaluation rollout to a rendered replay gallery."""

import copy
import json
import os
import struct
import sys
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from pufferlib import pufferl


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")
SCENARIO_LENGTH = 400
REPLAY_SEED = 1234
DISCRETE_ACTION_COUNT = 12
OBSERVATION_DIMENSION = 57
MIN_REPLAY_BYTES = 1_000
MIN_SCENARIO_HTML_BYTES = 50_000


class _ZeroPolicy:
    def eval(self):
        return self

    def forward_eval(self, observations):
        batch_size = observations.shape[0]
        logits = torch.zeros((batch_size, DISCRETE_ACTION_COUNT), device=observations.device)
        values = torch.zeros((batch_size, 1), device=observations.device)
        return (logits,), values


def _build_eval_args():
    with patch.object(sys, "argv", ["pufferl.py"]):
        args = pufferl.load_config("puffer_drive")

    args["train"].update(
        {
            "seed": REPLAY_SEED,
            "device": "cpu",
            "compile": False,
            "amp": False,
        }
    )
    args["vec"].update({"seed": REPLAY_SEED, "num_envs": 1})
    args["env"].update(
        {
            "num_agents": 1,
            "min_agents_per_env": 1,
            "max_agents_per_env": 1,
            "num_maps": 1,
            "map_dir": MAP_DIR,
            "action_type": "discrete",
            "dynamics_model": "jerk",
            "simulation_mode": "replay",
            "control_mode": "control_sdc_only",
            "sdc_controller": "replay",
            "non_sdc_controller": "replay",
            "scenario_length": SCENARIO_LENGTH,
            "resample_frequency": SCENARIO_LENGTH,
            "termination_mode": 0,
            "terminate_on_goal": True,
            "num_goals": 3,
            "goal_radius": 2.0,
            "goal_source": "gt",
            "obs_slots_partners_n": 1,
            "obs_slots_lane_n": 1,
            "obs_slots_boundary_n": 1,
            "obs_slots_traffic_controls_n": 1,
            "compute_eval_metrics": True,
        }
    )
    args["eval"]["observation_replay_writer_count"] = 1
    return args


def _read_replay_header(replay_path):
    payload = zlib.decompress(Path(replay_path).read_bytes())
    header_length = struct.unpack_from("<I", payload)[0]
    return json.loads(payload[4 : 4 + header_length])


@pytest.mark.parametrize("capture_observations", [False, True])
def test_validation_replay_html_generation(tmp_path, capture_observations):
    """A real discrete rollout produces a replay and usable HTML with optional observations."""
    assert os.path.isdir(MAP_DIR), f"Replay fixture missing: {MAP_DIR}"
    args = _build_eval_args()
    worker_env_kwargs = copy.deepcopy(args["env"])
    worker_env_kwargs.update(
        {
            "eval_mode": 1,
            "num_eval_scenarios": 1,
            "eval_map_indices": [0],
            "eval_scenario_seeds": [REPLAY_SEED],
            "starting_map": 0,
            "capture_replay": True,
            "replay_worker_idx": 0,
        }
    )
    replay_output_dir = tmp_path / "replays"

    summaries = pufferl._run_eval_rollout(
        args=args,
        env_name="puffer_drive",
        worker_env_kwargs=[worker_env_kwargs],
        total_steps=SCENARIO_LENGTH,
        desc="Replay HTML smoke test",
        expected_episodes=1,
        policy=_ZeroPolicy(),
        replay_output_dir=replay_output_dir,
        capture_observations=capture_observations,
    )

    assert len(summaries) == 1
    replay_path = summaries[0]["replay_path"]
    assert os.path.isfile(replay_path)
    assert os.path.getsize(replay_path) > MIN_REPLAY_BYTES

    replay_header = _read_replay_header(replay_path)
    assert replay_header["action_type"] == "discrete"
    assert replay_header["active_count"] == 1
    assert replay_header["chunks"]["policy_probs"]["shape"][1:] == [1, DISCRETE_ACTION_COUNT]
    if capture_observations:
        assert replay_header["obs_dim"] == OBSERVATION_DIMENSION
        assert replay_header["chunks"]["obs"]["shape"][1:] == [1, OBSERVATION_DIMENSION]
    else:
        assert replay_header["obs_dim"] == 0
        assert "obs" not in replay_header["chunks"]

    render_dir = pufferl._render_eval_replays(summaries, str(tmp_path))
    scenario_html_paths = [
        os.path.join(render_dir, filename)
        for filename in os.listdir(render_dir)
        if filename.endswith(".html") and filename != "index.html"
    ]

    assert len(scenario_html_paths) == 1
    assert os.path.getsize(scenario_html_paths[0]) > MIN_SCENARIO_HTML_BYTES
    assert os.path.isfile(os.path.join(render_dir, "index.html"))
