"""WOSAC inputs for replay scenes: getter layout, context (non-controlled) agents, and context in the metrics.

Run: python -m pytest tests/unit_tests/test_wosac_replay.py
"""

import os

import numpy as np
import pytest
import yaml

import pufferlib
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.evaluation_utils import evaluation_utils
from pufferlib.ocean.evaluation_utils.wosac.evaluator import WOSACEvaluator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUPLAN_MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "nuplan")
SCENARIO_LENGTH = 200
REPLAY_STEPS = 30
POSITION_TOLERANCE_M = 1e-3


def _make_env(control_mode, scene_copies):
    return Drive(
        num_agents=128 * scene_copies,
        min_agents_per_env=1,
        max_agents_per_env=128,
        num_maps=1,
        map_dir=NUPLAN_MAP_DIR,
        simulation_mode="replay",
        control_mode=control_mode,
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=SCENARIO_LENGTH,
        eval_mode=1,
        num_eval_scenarios=scene_copies,
        eval_map_indices=[0] * scene_copies,
        eval_scenario_seeds=[0] * scene_copies,
        collision_behavior="ignore",
        offroad_behavior="ignore",
        traffic_light_behavior="ignore",
        goal_source="gt",
    )


def test_ground_truth_rows_match_agents_and_carry_scene_index():
    env = _make_env("control_vehicles", scene_copies=2)
    env.reset(seed=0)
    controlled_row_count = int(env.agent_offsets[-1])
    ground_truth = env.get_ground_truth_trajectories()
    state = env.get_global_agent_state()

    assert ground_truth["x"].shape[-1] == SCENARIO_LENGTH
    valid_at_start = ground_truth["valid"][:controlled_row_count, 0, 0] == 1
    assert valid_at_start.sum() > 1
    start_error_m = np.abs(ground_truth["x"][:controlled_row_count, 0, 0] - state["x"][:controlled_row_count])
    assert start_error_m[valid_at_start].max() < POSITION_TOLERANCE_M

    expected_scene_ids = np.repeat(np.arange(env.num_envs), np.diff(env.agent_offsets))
    np.testing.assert_array_equal(ground_truth["scenario_id"][:controlled_row_count, 0], expected_scene_ids)
    road_edges = env.get_road_edge_polylines()
    edges_per_scene = [int((road_edges["scenario_id"] == scene_id).sum()) for scene_id in range(env.num_envs)]
    assert edges_per_scene[0] > 0 and edges_per_scene[0] == edges_per_scene[1]
    env.close()


def test_sdc_only_context_agents_are_the_replay_agents_at_their_logged_poses():
    env = _make_env("control_sdc_only", scene_copies=1)
    env.reset(seed=0)
    for _ in range(REPLAY_STEPS):
        env.step(np.zeros_like(env.actions))

    context = env.get_context_agent_trajectories()
    scene = env.get_state()[0]
    replay_agents = [scene["agents"][agent_idx] for agent_idx in scene["static_agent_indices"]]
    assert len(replay_agents) == len(scene["agents"]) - 1 == context["x"].shape[0]
    np.testing.assert_array_equal(context["scenario_id"][:, 0], 0)

    # get_state is in the scene's centered frame and the getters in world frame, so the gap is one constant offset.
    valid_now = context["valid"][:, 0, REPLAY_STEPS] == 1
    assert valid_now.sum() > 1
    offsets_x = context["x"][:, 0, REPLAY_STEPS] - np.array([agent["sim_x"] for agent in replay_agents])
    offsets_y = context["y"][:, 0, REPLAY_STEPS] - np.array([agent["sim_y"] for agent in replay_agents])
    assert np.ptp(offsets_x[valid_now]) < POSITION_TOLERANCE_M
    assert np.ptp(offsets_y[valid_now]) < POSITION_TOLERANCE_M
    assert np.all(context["length"][valid_now] > 0)

    # Replay agents sit on their logs, so the per-step state getter matches the logged window.
    context_state = env.get_context_agent_state()
    np.testing.assert_array_equal(context_state["valid"], context["valid"][:, 0, REPLAY_STEPS])
    assert np.abs(context_state["x"] - context["x"][:, 0, REPLAY_STEPS])[valid_now].max() < POSITION_TOLERANCE_M
    env.close()


def test_self_play_context_holds_the_uncontrolled_agents_and_rows_name_their_agents():
    env = _make_env("control_vehicles", scene_copies=1)
    env.reset(seed=0)
    controlled_row_count = int(env.agent_offsets[-1])
    controlled_agent_indices = env.get_controlled_agent_indices()[:controlled_row_count]
    scene = env.get_state()[0]

    assert controlled_agent_indices[0] == binding.EGO_IDX
    assert np.all(np.diff(controlled_agent_indices) > 0)
    assert env.get_context_agent_state()["x"].shape[0] == len(scene["agents"]) - controlled_row_count
    env.close()


def _straight_trajectories(start_x_m, num_agents, num_rollouts, num_steps):
    x = np.broadcast_to(start_x_m[:, None, None] + np.arange(num_steps) * 1.0, (num_agents, num_rollouts, num_steps))
    zeros = np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32)
    return {"x": x.astype(np.float32).copy(), "y": zeros.copy(), "z": zeros.copy(), "heading": zeros.copy()}


def test_context_agents_enter_interaction_features_only():
    num_scenes, num_rollouts, num_steps = 2, 4, 20
    evaluator = WOSACEvaluator(
        {
            "env": {"scenario_length": num_steps},
            "train": {"device": "cpu"},
            "eval": {"wosac_num_rollouts": num_rollouts},
        }
    )
    ground_truth = _straight_trajectories(np.zeros(num_scenes), num_scenes, 1, num_steps)
    ground_truth.update(
        valid=np.ones((num_scenes, 1, num_steps), dtype=np.int32),
        id=np.full((num_scenes, 1), -1, dtype=np.int32),
        scenario_id=np.arange(num_scenes, dtype=np.int32)[:, None],
    )
    simulated = _straight_trajectories(np.zeros(num_scenes), num_scenes, num_rollouts, num_steps)
    simulated["id"] = np.full((num_scenes, num_rollouts, num_steps), -1, dtype=np.int32)
    agent_state = {"length": np.full(num_scenes, 4.5, np.float32), "width": np.full(num_scenes, 2.0, np.float32)}
    road_edges = {
        "x": np.array([-50.0, 50.0, -50.0, 50.0], dtype=np.float32),
        "y": np.array([30.0, 30.0, 30.0, 30.0], dtype=np.float32),
        "lengths": np.array([2, 2], dtype=np.int32),
        "scenario_id": np.array([0, 1], dtype=np.int32),
    }
    # One logged car riding on top of scene 0's SDC; scene 1 has no context agent.
    context = _straight_trajectories(np.zeros(1), 1, 1, num_steps)
    context.update(
        valid=np.ones((1, 1, num_steps), dtype=np.int32),
        length=np.array([4.5], dtype=np.float32),
        width=np.array([2.0], dtype=np.float32),
        scenario_id=np.zeros((1, 1), dtype=np.int32),
    )
    scored = np.ones(num_scenes, dtype=bool)

    with_context = evaluator.compute_metrics(
        ground_truth, simulated, agent_state, road_edges, eval_mask=scored, context_trajectories=context
    )
    without_context = evaluator.compute_metrics(ground_truth, simulated, agent_state, road_edges, eval_mask=scored)

    assert with_context.loc[0, "num_collisions_sim"] == num_rollouts
    assert with_context.loc[0, "num_collisions_ref"] == 1
    assert with_context.loc[1, "num_collisions_sim"] == 0
    assert without_context["num_collisions_sim"].sum() == 0
    assert list(with_context["num_agents"]) == [1, 1]


def test_wosac_block_rejects_unknown_scored_agents(tmp_path):
    config_path = tmp_path / "wosac.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "env": {},
                "benchmarks": [
                    {
                        "name": "selfplay_wosac",
                        "seed": 0,
                        "num_scenarios": 1,
                        "wosac": {
                            "num_rollouts": 2,
                            "scenes_per_batch": 1,
                            "score_horizons_steps": [91],
                            "scored_agents": ["pedestrians"],
                        },
                        "env": {
                            "simulation_mode": "replay",
                            "control_mode": "control_vehicles",
                            "max_agents_per_env": 128,
                            "num_maps": 1,
                            "map_dir": NUPLAN_MAP_DIR,
                        },
                    }
                ],
            }
        )
    )
    with pytest.raises(pufferlib.APIUsageError, match="scored_agents"):
        evaluation_utils.load_benchmark_config(str(config_path), "selfplay_wosac")
