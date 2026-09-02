"""Fill the human-driver ghost (logged ego trajectory) into the obs_html/<token>.replay.zlib files of a
nuPlan run that was recorded before the planner wrote it, then re-render with render_obs_html.py.

usage: python scripts/eval/add_nuplan_human_ghost.py <group_dir> [--data-root DIR] [--maps-root DIR] [--tokens tok ...]

Needs the nuPlan devkit (carl_nuplan venv) and the split's log .db files under
<data-root>/nuplan-v1.1/splits/val. Scenarios are rebuilt with the run's own hydra scenario mapping /
vehicle parameters; the bin-frame origin is recovered from the replay's frame-0 ego (synced from nuPlan).
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from pufferlib import viz  # noqa: E402
from pufferlib.ocean.cosim import nuplan_bridge as nb  # noqa: E402

EGO_HEADING_IDX = 3  # agent_f32 columns: x, y, z, heading, ...
FRAME0_HEADING_TOL_RAD = 1e-3  # replay frame 0 is the nuPlan initial ego state, synced verbatim


def token_log_names(group_dir):
    rows = {}
    for csv in glob.glob(f"{group_dir}/simulation/*/20*/aggregator_metric/*_weighted_average_metrics_*.csv"):
        agg = pd.read_csv(csv)
        for _, row in agg[agg["scenario_type"] != "final_score"].iterrows():
            rows[str(row["scenario"])] = str(row["log_name"])
    return rows


def run_config(group_dir):
    configs = glob.glob(f"{group_dir}/simulation/*/20*/code/hydra/config.yaml")
    if not configs:
        raise SystemExit(f"no code/hydra/config.yaml under {group_dir}")
    return yaml.safe_load(open(configs[0]))


def build_scenarios(cfg, data_root, maps_root, wanted):
    """token -> NuPlanScenario for every (token, log_name) in `wanted`, extracted like the run did."""
    from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    builder_cfg = cfg["scenario_builder"]
    mapping_cfg = builder_cfg["scenario_mapping"]
    vehicle_cfg = {k: v for k, v in builder_cfg["vehicle_parameters"].items() if not k.startswith("_")}
    log_names = sorted(set(wanted.values()))
    db_files = [str(Path(data_root) / "nuplan-v1.1" / "splits" / "val" / f"{log_name}.db") for log_name in log_names]
    missing = [db for db in db_files if not os.path.exists(db)]
    if missing:
        raise SystemExit(f"{len(missing)} log db(s) missing locally, e.g. {missing[0]}")
    builder = NuPlanScenarioBuilder(
        data_root=str(Path(data_root) / "nuplan-v1.1" / "splits" / "val"),
        map_root=str(maps_root),
        sensor_root=None,
        db_files=db_files,
        map_version=builder_cfg["map_version"],
        include_cameras=False,
        verbose=False,
        scenario_mapping=ScenarioMapping(mapping_cfg["scenario_map"], mapping_cfg.get("subsample_ratio_override")),
        vehicle_parameters=VehicleParameters(**vehicle_cfg),
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=sorted(wanted.keys()),
        log_names=log_names,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
    )
    return {scenario.token: scenario for scenario in builder.get_scenarios(scenario_filter, Sequential())}


def replay_ghost(replay_path, scenario):
    """(frames, 1, 5) ghost in the replay's bin frame, or None with a reason when the replay can't be matched."""
    header, chunks = viz.read_replay_zlib(replay_path)
    if header["active_count"] != 1:
        return None, f"active_count {header['active_count']} != 1"
    ego0 = chunks["agent_f32"][0, 0]
    initial = scenario.get_ego_state_at_iteration(0)
    heading_delta = float(initial.center.heading) - float(ego0[EGO_HEADING_IDX])
    heading_error = abs((heading_delta + np.pi) % (2.0 * np.pi) - np.pi)
    if heading_error > FRAME0_HEADING_TOL_RAD:
        return None, f"frame-0 heading mismatch {heading_error:.4f} rad"
    origin = nb.NuPlanTransform(float(initial.center.x) - float(ego0[0]), float(initial.center.y) - float(ego0[1]))
    boxes = nb.logged_ego_boxes(scenario, origin)
    frames = int(header["frames"])
    ghost = np.zeros((frames, 1, 5), np.float32)
    count = min(frames, len(boxes))
    ghost[:count, 0] = boxes[:count]
    return ghost, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group_dir")
    ap.add_argument("--data-root", default=os.environ.get("NUPLAN_DATA_ROOT"))
    ap.add_argument("--maps-root", default=os.environ.get("NUPLAN_MAPS_ROOT"))
    ap.add_argument("--tokens", nargs="*", default=[])
    args = ap.parse_args()
    if not args.data_root or not args.maps_root:
        raise SystemExit("--data-root / --maps-root (or NUPLAN_DATA_ROOT / NUPLAN_MAPS_ROOT) required")
    obs_dir = Path(args.group_dir) / "obs_html"
    replays = {p.name[: -len(".replay.zlib")]: p for p in obs_dir.glob("*.replay.zlib")}
    if args.tokens:
        replays = {token: path for token, path in replays.items() if token in set(args.tokens)}
    if not replays:
        raise SystemExit(f"no obs_html/*.replay.zlib under {args.group_dir}")
    log_names = token_log_names(args.group_dir)
    wanted = {token: log_names[token] for token in replays if token in log_names}
    scenarios = build_scenarios(run_config(args.group_dir), args.data_root, args.maps_root, wanted)
    written, skipped = 0, []
    for token, path in sorted(replays.items()):
        scenario = scenarios.get(token)
        if scenario is None:
            skipped.append(f"{token}: no scenario ({'no log_name' if token not in wanted else 'not built'})")
            continue
        ghost, reason = replay_ghost(path, scenario)
        if ghost is None:
            skipped.append(f"{token}: {reason}")
            continue
        viz.set_replay_ghost(path, ghost)
        written += 1
    for line in skipped:
        print(f"[add_nuplan_human_ghost] skipped {line}")
    print(f"[add_nuplan_human_ghost] {written}/{len(replays)} replays updated -> now run scripts/eval/render_obs_html.py {args.group_dir}")


if __name__ == "__main__":
    main()
