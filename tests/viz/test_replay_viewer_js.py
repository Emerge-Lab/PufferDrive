import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import pufferlib.viz
from pufferlib import pufferl, replay_format, replay_video
from pufferlib.ocean.evaluation_utils import eval_replay as drive_eval_replay
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = Path(__file__).with_name("replay_viewer_harness.mjs")
CARLA_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/carla"
SEED = 42
SCENARIO_SEED = 4321
VIEWER_SCENARIO_LENGTH = 16
VIEWER_AGENT_COUNT = 8
VIEWER_PARTNER_SLOTS = 4
TARGET_FRAME = 5
DISCRETE_CLASSIC_ACTION_COUNT = 63
CAMERA_WIDTH_PX = 640
CAMERA_HEIGHT_PX = 360
MIN_PROBE_DEPTH_M = 1.0
ABSENT_AGENT_ID = 987654
OLD_REPLAY_FORMAT_VERSION = 1
NODE_TIMEOUT_SECONDS = 900
DECODE_TOLERANCE_M = 1e-9
HEADING_TOLERANCE_RAD = 1e-9
PROJECTION_TOLERANCE_PX = 1e-6
DEPTH_TOLERANCE_M = 1e-9
CAMERA_VECTOR_TOLERANCE = 1e-9
# probe = (forward_m, left_m, z_m) around the target agent
CAMERA_PROBES = [
    (0.0, 0.0, 0.0),
    (40.0, 0.0, 1.0),
    (10.0, 5.0, 0.0),
    (10.0, -5.0, 0.0),
    (60.0, 20.0, 0.0),
    (5.0, 0.0, 1.6),
    (-3.0, 2.0, 0.5),
    (30.0, -12.0, 2.0),
    (80.0, 0.0, 0.0),
    (-30.0, 0.0, 0.0),
    (150.0, -40.0, 0.0),
    (2.0, 1.0, 1.2),
]
VERBATIM_INVARIANTS = [
    'class="payload-chunk"',
    'id="reward-grid"',
    '"return (cum)"',
    "ctx.arc(g.x,g.y,g.radius",
    'const lattice = H.action_type === "lattice"',
]
NEW_ELEMENT_IDS = [
    "agentViewBtn",
    "agent-view-box",
    "agent-view-canvas",
    "agentViewPresetBtn",
    "agentViewExpandBtn",
    "view-pill",
]

requires_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


class ZeroPolicy(torch.nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.action_count = action_count

    def forward_eval(self, observations):
        batch_size = observations.shape[0]
        logits = torch.zeros((batch_size, self.action_count), device=observations.device)
        values = torch.zeros((batch_size, 1), device=observations.device)
        return (logits,), values


def _viewer_args():
    with patch.object(sys, "argv", ["pufferl.py"]):
        args = pufferl.load_config("puffer_drive")
    args["train"].update({"seed": SEED, "device": "cpu", "compile": False})
    args["vec"].update({"seed": SEED, "num_envs": 1, "num_workers": 1})
    args["env"].update(
        {
            "num_agents": VIEWER_AGENT_COUNT,
            "min_agents_per_env": VIEWER_AGENT_COUNT,
            "max_agents_per_env": VIEWER_AGENT_COUNT,
            "num_maps": 1,
            "map_dir": str(CARLA_MAP_DIR),
            "use_map_cache": True,
            "scenario_length": VIEWER_SCENARIO_LENGTH,
            "resample_frequency": VIEWER_SCENARIO_LENGTH,
            "action_type": "discrete",
            "dynamics_model": "classic",
            "termination_mode": False,
            "collision_behavior": "ignore",
            "offroad_behavior": "ignore",
            "obs_slots_partners_n": VIEWER_PARTNER_SLOTS,
            "obs_slots_lane_n": 16,
            "obs_slots_boundary_n": 16,
            "obs_slots_traffic_controls_n": 2,
            "obs_dropout_lane": 0.0,
            "obs_dropout_boundary": 0.0,
            "partner_blindness_prob": 0.0,
            "partner_blindness_trigger_prob": 0.0,
            "phantom_braking_prob": 0.0,
            "phantom_braking_trigger_prob": 0.0,
        }
    )
    args["eval"]["observation_replay_writer_count"] = 1
    args["eval"]["action_selection"] = "mode"
    return args


def _agent_pose(chunks, frame, agent_idx):
    state = chunks["agent_f32"][frame, agent_idx]
    return (
        float(state[replay_format.AGENT_F32_X_IDX]),
        float(state[replay_format.AGENT_F32_Y_IDX]),
        float(state[replay_format.AGENT_F32_HEADING_IDX]),
        float(state[replay_format.AGENT_F32_Z_IDX]),
    )


def _valid_agent_indices(chunks, frame):
    valid = chunks["agent_i32"][frame, :, replay_format.AGENT_I32_VALID_IDX] == 1
    return [int(agent_idx) for agent_idx in np.flatnonzero(valid)]


def _to_world_rows(points_m, x_m, y_m, heading_rad):
    return replay_format.ego_frame_to_world(points_m, x_m, y_m, heading_rad).reshape(-1, 4)


def _observed_world(header, chunks, frame, agent_idx):
    x_m, y_m, heading_rad, _ = _agent_pose(chunks, frame, agent_idx)
    slot_idx = int(chunks["agent_i32"][frame, agent_idx, replay_format.AGENT_I32_SLOT_IDX])
    decoded = replay_format.decode_observation(header, chunks["obs"][frame, slot_idx])
    partner_world = replay_format.ego_frame_to_world(decoded.partner_xy_m, x_m, y_m, heading_rad).reshape(-1, 2)
    tolerance_m = replay_format.REPLAY_VIEW_STYLE["partner_match_tolerance_m"]
    candidates = []
    for partner_idx, partner_slot in enumerate(decoded.partner_slot_idx):
        for other_idx in _valid_agent_indices(chunks, frame):
            if other_idx == agent_idx:
                continue
            other_x_m, other_y_m, _, _ = _agent_pose(chunks, frame, other_idx)
            distance_m = math.hypot(
                partner_world[partner_idx, 0] - other_x_m, partner_world[partner_idx, 1] - other_y_m
            )
            if distance_m <= tolerance_m:
                candidates.append((distance_m, int(partner_slot), other_idx, partner_idx))
    matched_partners, matched_agents = set(), []
    for _, _, other_idx, partner_idx in sorted(candidates):
        if partner_idx in matched_partners or other_idx in matched_agents:
            continue
        matched_partners.add(partner_idx)
        matched_agents.append(other_idx)
    unmatched = [
        {
            "x": float(partner_world[partner_idx, 0]),
            "y": float(partner_world[partner_idx, 1]),
            "h": heading_rad + float(decoded.partner_heading_rad[partner_idx]),
            "l": float(decoded.partner_length_m[partner_idx]),
            "w": float(decoded.partner_width_m[partner_idx]),
        }
        for partner_idx in range(len(decoded.partner_slot_idx))
        if partner_idx not in matched_partners
    ]
    stop_lines = _to_world_rows(decoded.stop_line_xy_m, x_m, y_m, heading_rad)
    return {
        "lanes": _to_world_rows(decoded.lane_segment_xy_m, x_m, y_m, heading_rad).tolist(),
        "bounds": _to_world_rows(decoded.boundary_segment_xy_m, x_m, y_m, heading_rad).tolist(),
        "stopLines": [
            {"x0": line[0], "y0": line[1], "x1": line[2], "y1": line[3], "type": int(kind), "state": int(state)}
            for line, kind, state in zip(stop_lines.tolist(), decoded.stop_line_type, decoded.stop_line_state)
        ],
        "partnerAgentIdx": sorted(matched_agents),
        "unmatchedPartners": unmatched,
        "partnerBlind": bool(chunks["agent_i32"][frame, agent_idx, replay_format.AGENT_I32_BLIND_IDX] == 1),
    }


def _choose_target(header, chunks):
    frame = min(TARGET_FRAME, header["frames"] - 1)
    valid_agents = _valid_agent_indices(chunks, frame)
    best = None
    for agent_idx in valid_agents:
        if chunks["agent_i32"][frame, agent_idx, replay_format.AGENT_I32_SLOT_IDX] < 0:
            continue
        observed = _observed_world(header, chunks, frame, agent_idx)
        if not (observed["lanes"] and observed["bounds"] and observed["partnerAgentIdx"]):
            continue
        unobserved_count = len(valid_agents) - 1 - len(observed["partnerAgentIdx"])
        score = (unobserved_count > 0, bool(observed["stopLines"]), len(observed["partnerAgentIdx"]))
        if best is None or score > best[0]:
            best = (score, agent_idx, observed)
    assert best is not None, "no agent observes lanes, boundaries and a matched partner"
    return frame, best[1], best[2]


def _camera_references(chunks, frame, agent_idx):
    x_m, y_m, heading_rad, agent_z_m = _agent_pose(chunks, frame, agent_idx)
    forward = np.array([math.cos(heading_rad), math.sin(heading_rad)])
    left = np.array([-math.sin(heading_rad), math.cos(heading_rad)])
    references = {}
    for preset in ("chase", "driver"):
        camera = replay_video.agent_view_camera(
            x_m,
            y_m,
            heading_rad,
            replay_format.REPLAY_VIEW_STYLE["cameras"][preset],
            CAMERA_WIDTH_PX,
            CAMERA_HEIGHT_PX,
            agent_z_m,
        )
        points = np.array(
            [
                [*(np.array([x_m, y_m]) + forward_m * forward + left_m * left), z_m]
                for forward_m, left_m, z_m in CAMERA_PROBES
            ]
        )
        pixels, depth_m = replay_video.project_points(points, camera)
        keep = np.abs(depth_m) >= MIN_PROBE_DEPTH_M
        references[preset] = {
            "eye": camera.eye_m.tolist(),
            "forward": camera.forward.tolist(),
            "right": camera.right.tolist(),
            "up": camera.up.tolist(),
            "focalLengthPx": float(camera.focal_length_px),
            "horizonYPx": float(camera.horizon_y_px),
            "points": points[keep].tolist(),
            "projected": np.column_stack([pixels[keep], depth_m[keep]]).tolist(),
        }
    return references


def _render_variant(header, chunks, html_path, drop_observations=False, agent_i32=None, format_version=None):
    variant_header = {key: value for key, value in header.items() if key != "chunks"}
    variant_chunks = {name: np.array(chunk) for name, chunk in chunks.items()}
    if drop_observations:
        variant_chunks = {
            name: chunk for name, chunk in variant_chunks.items() if name != "obs" and not name.startswith("pool_")
        }
        variant_header["obs_dim"] = 0
    if agent_i32 is not None:
        variant_chunks["agent_i32"] = agent_i32
    if format_version is not None:
        variant_header["replay_format_version"] = format_version
    compressed = pufferlib.viz._pack_replay_binary(variant_header, variant_chunks)
    pufferlib.viz._render_interactive_replay_payload(compressed, str(html_path))
    return html_path


@pytest.fixture(scope="module")
def viewer_replay(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("viewer_replay")
    args = _viewer_args()
    worker_env_kwargs, total_steps = drive_benchmark._plan_failure_replay_workers(
        args, [(0, SCENARIO_SEED)], num_workers=1, scenario_length=VIEWER_SCENARIO_LENGTH
    )
    summaries = pufferl._run_eval_rollout(
        args,
        "puffer_drive",
        worker_env_kwargs,
        total_steps,
        "Viewer replay",
        expected_episodes=1,
        policy=ZeroPolicy(action_count=DISCRETE_CLASSIC_ACTION_COUNT),
        replay_output_dir=output_dir / drive_eval_replay.ZLIB_REPLAY_DIR_NAME,
        capture_observations=True,
    )
    render_dir = Path(drive_eval_replay._render_eval_replays(summaries, str(output_dir), keep_zlib_replays=True))
    pages = sorted(page for page in render_dir.glob("*.html") if page.name != "index.html")
    assert len(pages) == 1
    header, chunks = replay_format.decode_interactive_replay(Path(summaries[0]["replay_path"]).read_bytes())
    return {"output_dir": output_dir, "html_path": pages[0], "header": header, "chunks": chunks}


@pytest.fixture(scope="module")
def harness_report(viewer_replay):
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    output_dir = viewer_replay["output_dir"]
    header, chunks = viewer_replay["header"], viewer_replay["chunks"]
    frame, target_idx, observed = _choose_target(header, chunks)
    valid_agents = _valid_agent_indices(chunks, frame)
    static_idx = next(agent_idx for agent_idx in valid_agents if agent_idx != target_idx)
    blind_agent_i32 = np.array(chunks["agent_i32"])
    blind_agent_i32[:, target_idx, replay_format.AGENT_I32_BLIND_IDX] = 1
    blind_agent_i32[:, static_idx, replay_format.AGENT_I32_SLOT_IDX] = -1
    agent_ids = chunks["agent_i32"][frame, :, replay_format.AGENT_I32_ID_IDX]
    config = {
        "mainHtml": str(viewer_replay["html_path"]),
        "noObsHtml": str(_render_variant(header, chunks, output_dir / "no_obs.html", drop_observations=True)),
        "blindHtml": str(_render_variant(header, chunks, output_dir / "blind.html", agent_i32=blind_agent_i32)),
        "oldFormatHtml": str(
            _render_variant(header, chunks, output_dir / "old_format.html", format_version=OLD_REPLAY_FORMAT_VERSION)
        ),
        "frame": frame,
        "targetIdx": target_idx,
        "targetId": int(agent_ids[target_idx]),
        "staticIdx": static_idx,
        "staticId": int(agent_ids[static_idx]),
        "absentId": ABSENT_AGENT_ID,
        "cameraWidthPx": CAMERA_WIDTH_PX,
        "cameraHeightPx": CAMERA_HEIGHT_PX,
        "cameraRefs": _camera_references(chunks, frame, target_idx),
        "validAgents": [
            {
                "idx": agent_idx,
                "x": _agent_pose(chunks, frame, agent_idx)[0],
                "y": _agent_pose(chunks, frame, agent_idx)[1],
                "l": float(chunks["agent_f32"][frame, agent_idx, replay_format.AGENT_F32_LENGTH_IDX]),
            }
            for agent_idx in valid_agents
        ],
        "reportPath": str(output_dir / "harness_report.json"),
    }
    config_path = output_dir / "harness_config.json"
    config_path.write_text(json.dumps(config))
    result = subprocess.run(
        ["node", str(HARNESS_PATH), str(config_path)], capture_output=True, text=True, timeout=NODE_TIMEOUT_SECONDS
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(Path(config["reportPath"]).read_text())
    return {"report": report, "config": config, "observed": observed}


def _step(harness_report, name):
    report = harness_report["report"]
    assert name not in report["stepErrors"], report["stepErrors"].get(name)
    assert name in report["data"], f"step {name} did not run; pages={report['pages']}"
    return report["data"][name]


def _wrap_angle(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _partner_order(partner):
    return (partner["x"], partner["y"])


def test_viewer_html_keeps_verbatim_invariants(viewer_replay):
    html = viewer_replay["html_path"].read_text()
    for invariant in VERBATIM_INVARIANTS:
        assert invariant in html, invariant
    assert html.count("H.trajectory_baseline") == 2


def test_viewer_html_has_the_agent_view_and_pill_elements(viewer_replay):
    html = viewer_replay["html_path"].read_text()
    for element_id in NEW_ELEMENT_IDS:
        assert f'id="{element_id}"' in html, element_id
    pill_position = html.index('id="view-pill"')
    assert html.index('id="ui-layer"') < pill_position
    assert not html.index('id="hud-telemetry"') < pill_position < html.index('id="obs-container"')


def test_viewer_html_fills_every_placeholder(viewer_replay):
    assert re.findall(r"__[A-Z][A-Z0-9_]*__", viewer_replay["html_path"].read_text()) == []


@requires_node
def test_viewer_pages_load_without_errors(harness_report):
    pages = harness_report["report"]["pages"]
    assert set(pages) == {"main", "noObs", "blind", "oldFormat"}
    for name, page in pages.items():
        assert page["loaded"], (name, page["errors"])
        assert page["errors"] == [], (name, page["errors"])
    assert _step(harness_report, "headerVersion") == replay_format.REPLAY_FORMAT_VERSION


@requires_node
def test_agent_view_opens_draws_and_closes(harness_report):
    agent_view = _step(harness_report, "agentView")
    assert agent_view["onBefore"] is False
    assert agent_view["onAfterOpen"] is True
    assert agent_view["boxDisplay"] != "none"
    assert agent_view["openCalls"].get("fill", 0) > 0
    assert agent_view["openCalls"].get("stroke", 0) > 0
    assert agent_view["driverCallCount"] > 0
    assert agent_view["canvasWidth"] * 9 == agent_view["canvasHeight"] * 16
    assert agent_view["onAfterClose"] is False
    assert agent_view["callsAfterClose"] == 0


@requires_node
def test_agent_view_switches_preset_and_expands(harness_report, viewer_replay):
    viewer_html = viewer_replay["html_path"].read_text()
    agent_view = _step(harness_report, "agentView")
    assert agent_view["presets"] == ["chase", "driver", "chase"]
    assert agent_view["expandedClassAfterToggle"] is True
    assert agent_view["expandedClassAfterSecondToggle"] is False
    expanded_rule = re.search(r"#hud-telemetry\.expanded\s*\{([^}]*)\}", viewer_html)
    assert expanded_rule is not None and "680px" in expanded_rule.group(1) and "45vw" in expanded_rule.group(1)


@requires_node
def test_agent_view_applies_the_observed_filter(harness_report):
    agent_view = _step(harness_report, "agentView")
    assert agent_view["observedForAgentView"] is True
    assert 0 < agent_view["fillsWithV"] <= agent_view["fillsWithoutV"]


@requires_node
@pytest.mark.parametrize("preset", ["chase", "driver"])
def test_js_agent_view_camera_matches_python(harness_report, preset):
    reference = harness_report["config"]["cameraRefs"][preset]
    camera = _step(harness_report, "camera")[preset]["camera"]
    for key in ("eye", "forward", "right", "up"):
        np.testing.assert_allclose(camera[key], reference[key], rtol=0.0, atol=CAMERA_VECTOR_TOLERANCE, err_msg=key)
    assert camera["focalLengthPx"] == pytest.approx(reference["focalLengthPx"], abs=PROJECTION_TOLERANCE_PX)
    assert camera["horizonYPx"] == pytest.approx(reference["horizonYPx"], abs=PROJECTION_TOLERANCE_PX)
    assert camera["widthPx"] == CAMERA_WIDTH_PX
    assert camera["heightPx"] == CAMERA_HEIGHT_PX


@requires_node
@pytest.mark.parametrize("preset", ["chase", "driver"])
def test_js_cam_project_matches_python_project_points(harness_report, preset):
    reference = np.array(harness_report["config"]["cameraRefs"][preset]["projected"])
    projected = np.array(_step(harness_report, "camera")[preset]["projected"])
    assert projected.shape == reference.shape
    assert len(reference) >= len(CAMERA_PROBES) // 2
    np.testing.assert_allclose(projected[:, :2], reference[:, :2], rtol=0.0, atol=PROJECTION_TOLERANCE_PX)
    np.testing.assert_allclose(projected[:, 2], reference[:, 2], rtol=0.0, atol=DEPTH_TOLERANCE_M)


@requires_node
def test_js_decode_observed_world_matches_python_decode(harness_report):
    expected = harness_report["observed"]
    decoded = _step(harness_report, "decode")
    assert decoded is not None
    for key in ("lanes", "bounds"):
        actual = np.array(decoded[key], dtype=np.float64).reshape(-1, 4)
        reference = np.array(expected[key], dtype=np.float64).reshape(-1, 4)
        assert actual.shape == reference.shape, key
        np.testing.assert_allclose(actual, reference, rtol=0.0, atol=DECODE_TOLERANCE_M, err_msg=key)
    assert len(decoded["stopLines"]) == len(expected["stopLines"])
    for actual, reference in zip(decoded["stopLines"], expected["stopLines"]):
        for key in ("x0", "y0", "x1", "y1"):
            assert actual[key] == pytest.approx(reference[key], abs=DECODE_TOLERANCE_M), key
        assert (actual["type"], actual["state"]) == (reference["type"], reference["state"])
    assert sorted(decoded["partnerAgentIdx"]) == expected["partnerAgentIdx"]
    assert len(decoded["unmatchedPartners"]) == len(expected["unmatchedPartners"])
    for actual, reference in zip(
        sorted(decoded["unmatchedPartners"], key=_partner_order),
        sorted(expected["unmatchedPartners"], key=_partner_order),
    ):
        for key in ("x", "y", "l", "w"):
            assert actual[key] == pytest.approx(reference[key], abs=DECODE_TOLERANCE_M), key
        assert abs(_wrap_angle(actual["h"] - reference["h"])) <= HEADING_TOLERANCE_RAD
    assert decoded["partnerBlind"] is False


@requires_node
def test_keyv_toggles_observed_only_for_an_eligible_agent(harness_report):
    observed_only = _step(harness_report, "observedOnly")
    target_id = harness_report["config"]["targetId"]
    assert observed_only["offBefore"] is False
    assert observed_only["onAfterV"] is True
    assert "observed by agent" in observed_only["pillWhenOn"]["text"]
    assert str(target_id) in observed_only["pillWhenOn"]["text"]
    assert observed_only["pillWhenOn"]["visible"]
    assert "partner-blind" not in observed_only["pillWhenOn"]["text"]
    assert observed_only["offAfterSecondV"] is False
    pill_after = observed_only["pillAfterSecondV"]
    assert not (pill_after["visible"] and "observed by agent" in pill_after["text"])


@requires_node
def test_v_mode_draws_only_the_target_and_matched_partners(harness_report):
    observed_only = _step(harness_report, "observedOnly")
    config = harness_report["config"]
    all_valid = sorted(agent["idx"] for agent in config["validAgents"])
    expected_with_v = sorted([config["targetIdx"], *harness_report["observed"]["partnerAgentIdx"]])
    assert len(expected_with_v) < len(all_valid)
    assert sorted(observed_only["drawnWithoutV"]) == all_valid
    assert sorted(observed_only["drawnWithV"]) == expected_with_v


@requires_node
def test_v_mode_skips_the_global_map_paths_and_current_lane(harness_report):
    observed_only = _step(harness_report, "observedOnly")
    non_empty_paths = [op_count > 0 for op_count in observed_only["mapPathOpCounts"]]
    assert any(non_empty_paths)
    stroked_without_v = [
        stroked for stroked, non_empty in zip(observed_only["mapPathsStrokedWithoutV"], non_empty_paths) if non_empty
    ]
    assert stroked_without_v == [True] * sum(non_empty_paths)
    assert observed_only["mapPathsStrokedWithV"] == [False, False, False]
    assert observed_only["lanePathsStrokedWithV"] == 0


@requires_node
def test_escape_and_an_empty_click_turn_v_off(harness_report):
    escape = _step(harness_report, "escapeTurnsVOff")
    assert escape["onAfterV"] is True
    assert escape["afterEscape"] is False
    click = _step(harness_report, "emptyClickTurnsVOff")
    assert click["onAfterV"] is True
    assert click["point"] is not None, "no empty canvas corner to click"
    assert click["afterClick"] is False


@requires_node
def test_keyv_without_a_selection_stays_off(harness_report):
    assert _step(harness_report, "noSelection")["afterV"] is False


@requires_node
def test_v_pill_reports_an_absent_agent(harness_report):
    absent = _step(harness_report, "absentAgent")
    assert absent["before"] is False
    assert absent["afterV"] is False
    assert f"agent {ABSENT_AGENT_ID} not present" in absent["pill"]["text"]
    assert absent["pill"]["visible"]


@requires_node
def test_v_pill_reports_a_replay_without_observations(harness_report):
    no_observations = _step(harness_report, "noObs")
    assert no_observations["decode"] is None
    assert no_observations["afterV"] is False
    assert "V needs eval.capture_observations=true" in no_observations["pill"]["text"]
    assert no_observations["pill"]["visible"]


@requires_node
def test_v_pill_reports_an_agent_without_observation(harness_report):
    static_agent = _step(harness_report, "staticAgent")
    assert static_agent["slot"] < 0
    assert static_agent["afterV"] is False
    expected_text = f"agent {harness_report['config']['staticId']} has no observation"
    assert expected_text in static_agent["pill"]["text"]
    assert static_agent["pill"]["visible"]


@requires_node
def test_v_pill_marks_a_partner_blind_agent(harness_report):
    blind = _step(harness_report, "blind")
    assert blind["afterV"] is True
    assert "observed by agent" in blind["pill"]["text"]
    assert "partner-blind" in blind["pill"]["text"]
    assert blind["partnerBlind"] is True


@requires_node
def test_old_format_replay_is_not_decoded(harness_report):
    old_format = _step(harness_report, "oldFormat")
    assert old_format["mentionsOldFormat"] is True
    assert old_format["afterV"] is False
