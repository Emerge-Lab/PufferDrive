"""Build a nuBoard folder that contains only the failing scenarios of a nuPlan run, then launch nuBoard on it.

usage: python scripts/eval/nuboard_failures.py <group_dir> [--max-score 0.5] [--metric no_ego_at_fault_collisions ...]
       [--out <dir>] [--launch] [--port 5006]

nuBoard has no score filter of its own, so this mirrors <group_dir>/simulation/<challenge>/<ts>/ into
<out> with symlinks to the selected scenarios' simulation logs, metric/aggregator tables filtered to
those scenarios, and a .nuboard descriptor. --launch starts nuBoard on the mirror (Hydra
run_nuboard.py from $NUPLAN_DEVKIT_ROOT); logs written on another machine are loaded through
scripts/eval/nuplan_log_remap.py (set NUPLAN_DATA_ROOT / NUPLAN_MAPS_ROOT).

Selection: scenarios with aggregated score < --max-score, OR any --metric below 1.0.
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


def select_failures(sim_dir, max_score, metrics):
    agg = pd.read_csv(glob.glob(f"{sim_dir}/aggregator_metric/*_weighted_average_metrics_*.csv")[0])
    per_scenario = agg[(agg["scenario_type"] != "final_score") & (agg["scenario"] != agg["scenario_type"])]
    mask = per_scenario["score"] < max_score
    for metric in metrics:
        if metric not in per_scenario:
            raise SystemExit(f"unknown metric {metric}; available: {[c for c in per_scenario.columns if c not in ('scenario', 'scenario_type', 'log_name', 'planner_name', 'aggregator_type', 'num_scenarios')]}")
        mask |= per_scenario[metric] < 1.0
    return per_scenario[mask].sort_values("score")


def build_mirror(sim_dir, out_dir, selected):
    from nuplan.planning.nuboard.base.data_class import NuBoardFile

    sim_dir, out_dir = Path(sim_dir).resolve(), Path(out_dir).resolve()  # absolute: symlink targets must not be relative
    tokens = set(selected["scenario"].astype(str))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "simulation_log").mkdir(parents=True)
    linked = 0
    for log_path in glob.glob(f"{sim_dir}/simulation_log/**/*.xz", recursive=True):
        rel = Path(log_path).relative_to(sim_dir / "simulation_log")
        if rel.parent.name not in tokens:
            continue
        target = out_dir / "simulation_log" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(log_path, target)
        linked += 1
    (out_dir / "metrics").mkdir()
    for f in glob.glob(f"{sim_dir}/metrics/*.parquet"):
        df = pd.read_parquet(f)
        df[df["scenario_name"].astype(str).isin(tokens)].to_parquet(out_dir / "metrics" / Path(f).name)
    (out_dir / "aggregator_metric").mkdir()
    for f in glob.glob(f"{sim_dir}/aggregator_metric/*"):
        if not f.endswith((".parquet", ".csv")):
            continue
        df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        if "scenario" in df:  # per-scenario rows -> keep failures plus the per-type / final summary rows
            keep = df["scenario"].astype(str).isin(tokens)
            if "scenario_type" in df:
                keep |= (df["scenario"] == df["scenario_type"]) | (df["scenario_type"] == "final_score")
            df = df[keep]
        if f.endswith(".parquet"):
            df.to_parquet(out_dir / "aggregator_metric" / Path(f).name)
        else:
            df.to_csv(out_dir / "aggregator_metric" / Path(f).name, index=False)
    NuBoardFile(
        simulation_main_path=str(out_dir),
        metric_main_path=str(out_dir),
        metric_folder="metrics",
        aggregator_metric_folder="aggregator_metric",
        simulation_folder="simulation_log",
    ).save_nuboard_file(out_dir / "failures.nuboard")
    return linked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group_dir")
    ap.add_argument("--max-score", type=float, default=0.5)
    ap.add_argument("--metric", action="append", default=[], help="also select scenarios where this metric is < 1 (repeatable)")
    ap.add_argument("--out", default=None, help="mirror directory (default <group_dir>/nuboard_failures)")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--port", type=int, default=5006)
    args = ap.parse_args()
    sim_dirs = sorted(glob.glob(f"{args.group_dir}/simulation/*/20*"))
    if len(sim_dirs) != 1:
        raise SystemExit(f"expected exactly one simulation/<challenge>/<timestamp> under {args.group_dir}, found {sim_dirs}")
    selected = select_failures(sim_dirs[0], args.max_score, args.metric)
    out_dir = Path(args.out or f"{args.group_dir}/nuboard_failures").resolve()
    linked = build_mirror(sim_dirs[0], out_dir, selected)
    print(f"[nuboard_failures] {len(selected)} scenarios selected, {linked} logs linked -> {out_dir}")
    print(selected[["scenario", "scenario_type", "score"]].head(15).to_string(index=False))
    devkit = os.environ.get("NUPLAN_DEVKIT_ROOT", "")
    cmd = f"python {devkit}/nuplan/planning/script/run_nuboard.py simulation_path=[{out_dir}] port_number={args.port}"
    if not args.launch:
        print("launch with:\n  " + cmd)
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nuplan_log_remap  # noqa: F401  patches scenario/map paths for logs written elsewhere
    from nuplan.planning.simulation import simulation_log as _sl

    _sl.SimulationLog.load_data = classmethod(lambda cls, file_path: nuplan_log_remap.load_log(file_path))
    sys.argv = [sys.argv[0], f"simulation_path=[{out_dir}]", f"port_number={args.port}"]
    import runpy

    runpy.run_path(f"{devkit}/nuplan/planning/script/run_nuboard.py", run_name="__main__")


if __name__ == "__main__":
    main()
