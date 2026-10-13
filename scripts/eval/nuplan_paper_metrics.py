"""Score nuPlan co-sim runs with the HIFT-PPO paper's metrics (Appendix F.1).

  score-logs <group>                 ADE, FDE and red-light per scenario from simulation logs -> paper_log_metrics.csv
  table <challenge> <label=group>... one row per run: the paper's Table 1 columns

Run with the co-sim venv and the PufferDrive checkout on PYTHONPATH (the logs pickle the planner).
"""

import argparse
import glob
import lzma
import os
import pickle
from multiprocessing import Pool

import numpy as np

RED_LIGHT_BUFFER_M = 0.5
RED_LIGHT_ENTRY_MAX_PROGRESS_M = 3.0
LANE_HEADING_PROBE_M = 1.0
MIN_HEADING_ALIGNMENT_COS = float(np.cos(np.deg2rad(30.0)))
SPEED_LIMIT_NORM_MPS = 2.23
BRAKE_TAP_JERK_MPS3 = -3.7
SCORE_WORKER_COUNT = 16
LOG_METRICS_FILE = "paper_log_metrics.csv"


def load_simulation_log(path):
    import msgpack

    with open(path, "rb") as file:
        return pickle.loads(msgpack.unpackb(lzma.decompress(file.read())))


def followed_red_connector_crossed(map_api, ego, red_connector_ids, entry_was_red):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Point

    center = ego.center
    half_length_m = ego.car_footprint.length / 2
    front = Point(center.x + half_length_m * np.cos(center.heading), center.y + half_length_m * np.sin(center.heading))
    crossed = False
    for connector in map_api.get_all_map_objects(front, SemanticMapLayer.LANE_CONNECTOR):
        if not connector.has_traffic_lights():
            continue
        connector_id = str(connector.id)
        baseline = connector.baseline_path.linestring
        progress_m = baseline.project(front)
        ahead = baseline.interpolate(min(progress_m + LANE_HEADING_PROBE_M, baseline.length))
        behind = baseline.interpolate(max(progress_m - LANE_HEADING_PROBE_M, 0.0))
        lane_heading = np.arctan2(ahead.y - behind.y, ahead.x - behind.x)
        # crossing connectors overlap inside intersections; only the one the ego follows counts
        if np.cos(lane_heading - center.heading) < MIN_HEADING_ALIGNMENT_COS:
            continue
        is_red = connector_id in red_connector_ids
        entry_was_red.setdefault(connector_id, is_red and progress_m < RED_LIGHT_ENTRY_MAX_PROGRESS_M)
        crossed = crossed or (entry_was_red[connector_id] and is_red and progress_m > RED_LIGHT_BUFFER_M)
    return crossed


def score_log(path):
    from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

    log = load_simulation_log(path)
    scenario = log.scenario
    errors_m, entry_was_red, ran_red_light = [], {}, False
    for sample in log.simulation_history.data:
        ego = sample.ego_state
        expert = scenario.get_ego_state_at_iteration(sample.iteration.index)
        errors_m.append(float(np.hypot(ego.center.x - expert.center.x, ego.center.y - expert.center.y)))
        red_connector_ids = {
            str(light.lane_connector_id)
            for light in sample.traffic_light_status
            if light.status == TrafficLightStatusType.RED
        }
        ran_red_light |= followed_red_connector_crossed(scenario.map_api, ego, red_connector_ids, entry_was_red)
    return scenario.scenario_name, float(np.mean(errors_m)), errors_m[-1], int(ran_red_light)


def score_logs(group):
    for run_dir in sorted(glob.glob(f"{group}/simulation/*/*")):
        log_paths = sorted(glob.glob(f"{run_dir}/simulation_log/**/*.msgpack.xz", recursive=True))
        if not log_paths:
            continue
        with Pool(SCORE_WORKER_COUNT) as pool:
            rows = pool.map(score_log, log_paths)
        with open(os.path.join(run_dir, LOG_METRICS_FILE), "w") as file:
            file.write("scenario,ade,fde,red_light\n")
            for scenario_name, ade_m, fde_m, ran_red_light in rows:
                file.write(f"{scenario_name},{ade_m:.4f},{fde_m:.4f},{ran_red_light}\n")
        print(run_dir, len(rows), "scenarios scored")


def read_metric(run_dir, metric_name):
    import pandas as pd

    return pd.read_parquet(os.path.join(run_dir, "metrics", f"{metric_name}.parquet")).set_index("scenario_name")


def table_row(run_dir):
    import pandas as pd

    collisions = read_metric(run_dir, "no_ego_at_fault_collisions")
    drivable = read_metric(run_dir, "drivable_area_compliance")
    progress = read_metric(run_dir, "ego_progress_along_expert_route")
    over_speed = read_metric(run_dir, "speed_limit_compliance")["time_series_values"]
    lon_jerk = read_metric(run_dir, "ego_lon_jerk")["time_series_values"]
    row = {
        "n": len(collisions),
        "collision": 1 - collisions["metric_score"].mean(),
        "drivable": 1 - drivable["metric_score"].mean(),
        "speed": over_speed.apply(lambda v: 0.0 if v is None else min(1.0, float(np.mean(v)) / SPEED_LIMIT_NORM_MPS)).mean(),
        "progress": progress["ego_expert_progress_along_route_ratio_stat_value"].clip(0, 1).mean(),
        "brake_tap": lon_jerk.apply(lambda v: float(v is not None and np.min(v) < BRAKE_TAP_JERK_MPS3)).mean(),
    }
    log_metrics_path = os.path.join(run_dir, LOG_METRICS_FILE)
    log_metrics = pd.read_csv(log_metrics_path) if os.path.exists(log_metrics_path) else None
    for column in ("ade", "fde", "red_light"):
        row[column] = float("nan") if log_metrics is None else log_metrics[column].mean()
    return row


def print_table(challenge, labeled_groups):
    columns = ["ade", "fde", "collision", "drivable", "red_light", "speed", "progress", "brake_tap"]
    print("run".ljust(28), "n".rjust(4), *(column.rjust(10) for column in columns))
    for labeled_group in labeled_groups:
        label, group = labeled_group.split("=", 1)
        run_dirs = sorted(glob.glob(f"{group}/simulation/closed_loop_{challenge}_agents_pufferdrive/*"))
        if not run_dirs:
            print(label.ljust(28), "missing")
            continue
        row = table_row(run_dirs[-1])
        print(label.ljust(28), str(row["n"]).rjust(4), *(f"{row[column]:10.4f}" for column in columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("score-logs").add_argument("group")
    table_parser = commands.add_parser("table")
    table_parser.add_argument("challenge", choices=["nonreactive", "reactive"])
    table_parser.add_argument("labeled_groups", nargs="+")
    args = parser.parse_args()
    if args.command == "score-logs":
        score_logs(args.group)
    else:
        print_table(args.challenge, args.labeled_groups)
