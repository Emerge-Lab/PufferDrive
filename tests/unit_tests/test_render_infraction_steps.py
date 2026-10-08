"""Infraction -> replay step links in the obs replay panels of the CARLA and nuPlan renderers."""

import os
import sys

import numpy as np
import pandas as pd

from pufferlib import viz


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "eval")
)

import render_carla_obs_html as carla_render
import render_obs_html as nuplan_render


FRAMES, AGENT_CAP = 3000, 3
TICK_DT = 0.05
OFFSET = (10.0, 20.0)
COLLISION_FRAME = 600


def _replay(ego_xy, collision_frames=()):
    replay = {
        "env": {
            "num_goals": 2,
            "goal_radius": 2.0,
            "reward_conditioning": False,
            "obs_slots_partners_n": 4,
            "obs_slots_lane_n": 4,
            "obs_slots_boundary_n": 4,
            "obs_slots_traffic_controls_n": 1,
        },
        "agent_f32": np.zeros((FRAMES, AGENT_CAP, 13), np.float32),
        "agent_i32": np.zeros((FRAMES, AGENT_CAP, 10), np.int32),
        "metrics_f32": np.zeros((FRAMES, AGENT_CAP, len(viz.METRIC_LABELS)), np.float32),
        "puffer_f32": np.zeros((FRAMES, AGENT_CAP, 4), np.float32),
        "traffic_i16": np.zeros((FRAMES, 1, 3), np.int16),
        "raw_action": np.ones((FRAMES, 1, 2), np.float32),
        "clipped_action": np.ones((FRAMES, 1, 2), np.float32),
        "value": np.zeros((FRAMES, 1), np.float32),
        "entropy": np.zeros((FRAMES, 1), np.float32),
        "obs": None,
    }
    replay["agent_f32"][:, 0, :2] = ego_xy
    replay["metrics_f32"][list(collision_frames), 0, nuplan_render.COLLISION_METRIC_IDX] = 1.0
    return replay


def _straight_ego_xy():
    return np.column_stack([0.5 * np.arange(FRAMES), np.full(FRAMES, 100.0)])


def _carla_record():
    # bin (300, 100) = ego at frame 600; CARLA x = bin_x - tx, CARLA y = -(bin_y - ty)
    return {
        "route_id": "RouteScenario_3_rep0",
        "status": "Completed",
        "scores": {"score_composed": 50.0, "score_route": 100.0, "score_penalty": 0.5},
        "infractions": {
            "collisions_vehicle": [
                "Agent collided against object with type=vehicle and id=1 at (x=290.0, y=-80.0, z=0.0)"
            ],
            "route_dev": ["Agent deviated from the route at (x=5000.0, y=0.0, z=0.0)"],
            "outside_route_lanes": [
                "Agent went outside its route lanes for about 18.0 meters (1.5% of the completed route)"
            ],
        },
    }


def test_read_replay_zlib_chunk_subset_matches_full_read(tmp_path):
    path = tmp_path / "route.replay.zlib"
    viz.save_interactive_replay_zlib({}, _replay(_straight_ego_xy(), collision_frames=(7,)), str(path))
    header_full, chunks_full = viz.read_replay_zlib(path)
    header_part, chunks_part = viz.read_replay_zlib(path, chunk_names=("agent_f32", "metrics_f32"))
    assert header_part == header_full
    assert set(chunks_part) == {"agent_f32", "metrics_f32"}
    for name, arr in chunks_part.items():
        assert np.array_equal(arr, chunks_full[name])
    assert chunks_full["agent_f32"].nbytes > viz.REPLAY_HEADER_PROBE_BYTES


def test_carla_infraction_steps_link_nearest_ego_frame(tmp_path):
    path = tmp_path / "route.replay.zlib"
    viz.save_interactive_replay_zlib({}, _replay(_straight_ego_xy()), str(path))
    record = _carla_record()
    log_meta = {"offset": list(OFFSET), "tick_dt": TICK_DT}

    steps = carla_render.infraction_steps(record, str(path), "Town01", log_meta)
    collision, deviation, outside = (
        record["infractions"][key][0] for key in ("collisions_vehicle", "route_dev", "outside_route_lanes")
    )
    assert steps[collision] == viz.replay_step_link(COLLISION_FRAME, COLLISION_FRAME * TICK_DT)
    assert steps[deviation] == "beyond the replay"
    assert outside not in steps
    assert carla_render.infraction_steps(record, str(path), "", {}) == {}

    panel = carla_render.score_panel(record, "Town01", steps)
    assert f'onclick="step={COLLISION_FRAME};' in panel
    assert "beyond the replay" in panel


def test_carla_infraction_steps_fall_back_to_town_offset(tmp_path):
    path = tmp_path / "route.replay.zlib"
    viz.save_interactive_replay_zlib({}, _replay(_straight_ego_xy()), str(path))
    tx, ty = carla_render.TOWN_OFFSETS["Town01"]
    message = f"Agent ran a red light 12 at (x={300.0 - tx:.3f}, y={-(100.0 - ty):.3f}, z=0.0)"
    record = {"infractions": {"red_light": [message]}}
    steps = carla_render.infraction_steps(record, str(path), "Town01", {})
    assert steps[message] == viz.replay_step_link(COLLISION_FRAME)


def _write_series(metrics_dir, name, values, extra=None):
    row = {"scenario_name": "tok", "time_series_values": None if values is None else np.asarray(values, np.float64)}
    row.update(extra or {})
    pd.DataFrame([row]).to_parquet(metrics_dir / f"{name}.parquet")


def test_nuplan_violation_steps_from_metric_series(tmp_path):
    metrics_dir = tmp_path / "simulation" / "closed_loop" / "2026.01.01.00.00.00" / "metrics"
    metrics_dir.mkdir(parents=True)
    _write_series(metrics_dir, "corners_in_drivable_area", [1.0, 1.0, 0.0, 0.0])
    _write_series(metrics_dir, "driving_direction_compliance", [0.0, -1.0, -3.0])
    _write_series(metrics_dir, "time_to_collision_within_bound", [np.inf, 2.0, 0.9])
    _write_series(metrics_dir, "speed_limit_compliance", [0.0, 0.0, 0.0, 0.4])
    assert nuplan_render.violation_steps(str(tmp_path)) == {
        "tok": {
            "drivable_area_compliance": 2,
            "driving_direction_compliance": 2,
            "time_to_collision_within_bound": 2,
            "speed_limit_compliance": 3,
        }
    }


def test_nuplan_violation_steps_skip_compliant_and_missing_series(tmp_path):
    metrics_dir = tmp_path / "simulation" / "closed_loop" / "2026.01.01.00.00.00" / "metrics"
    metrics_dir.mkdir(parents=True)
    _write_series(metrics_dir, "speed_limit_compliance", [0.0, 0.0])
    _write_series(metrics_dir, "corners_in_drivable_area", None)
    assert nuplan_render.violation_steps(str(tmp_path)) == {}


def test_nuplan_comfort_failures_carry_first_out_of_bound_step(tmp_path):
    metrics_dir = tmp_path / "simulation" / "closed_loop" / "2026.01.01.00.00.00" / "metrics"
    metrics_dir.mkdir(parents=True)
    stats = {
        "abs_ego_lon_jerk_within_bounds_stat_value": False,
        "min_ego_lon_jerk_stat_value": 1.0,
        "max_ego_lon_jerk_stat_value": 5.0,
    }
    _write_series(metrics_dir, "ego_lon_jerk", [1.0, 5.0, 1.0], stats)
    comfort = nuplan_render.comfort_failures(str(tmp_path))
    assert comfort == {
        "tok": [f"ego_lon_jerk 1.00..5.00 (bound |j| <= 4.13 m/s^3) &middot; {nuplan_render.step_link(1)}"]
    }


def test_nuplan_replay_collision_step(tmp_path):
    path = tmp_path / "tok.replay.zlib"
    viz.save_interactive_replay_zlib({}, _replay(np.zeros((FRAMES, 2)), collision_frames=(40, 41)), str(path))
    assert nuplan_render.replay_collision_step(str(path)) == 40
    viz.save_interactive_replay_zlib({}, _replay(np.zeros((FRAMES, 2))), str(path))
    assert nuplan_render.replay_collision_step(str(path)) is None


def test_nuplan_score_panel_links_violated_metrics_only():
    row = pd.Series(
        {metric: 1.0 for metric, _, _ in nuplan_render.SCORED_METRICS} | {"score": 0.4, "scenario_type": "t"}
    )
    row["speed_limit_compliance"] = 0.8
    row[nuplan_render.COLLISION_METRIC] = 0.0
    steps = {"speed_limit_compliance": 12, nuplan_render.COLLISION_METRIC: 40, "drivable_area_compliance": 3}
    panel = nuplan_render.score_panel("tok", row, [], steps)
    assert 'onclick="step=12;' in panel
    assert 'onclick="step=40;' in panel and "(shadow env flag)" in panel
    assert 'onclick="step=3;' not in panel
