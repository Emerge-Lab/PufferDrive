"""Where the files of a nuPlan closed-loop run of the PufferDrive planner live. A fresh run is nuPlan's own tree,
simulation/<challenge>/<timestamp>/{aggregator_metric,metrics,simulation_log}; after scripts/eval/tidy_nuplan_run.py
the run holds overview.csv, scenarios.csv, obs_html/ and details/{aggregated_metrics.csv,metrics,simulation_log}.
Every lookup raises FileNotFoundError when the run has neither layout."""

import glob
from pathlib import Path

DETAILS_DIR = "details"
AGGREGATED_METRICS_CSV = "aggregated_metrics.csv"
SIMULATION_LOG_SUFFIX = ".msgpack.xz"


def raw_simulation_dir(group_dir):
    """-> nuPlan's simulation/<challenge>/<timestamp> directory of an untidied run, None when there is none."""
    matches = sorted(glob.glob(f"{group_dir}/simulation/*/20*"))
    if len(matches) > 1:
        raise FileNotFoundError(f"expected one simulation/<challenge>/<timestamp> under {group_dir}, found {matches}")
    return Path(matches[0]) if matches else None


def details_dir(group_dir):
    """-> the directory holding metrics/ and simulation_log/: details/ once tidied, else nuPlan's simulation dir."""
    tidied = Path(group_dir) / DETAILS_DIR
    if tidied.is_dir():
        return tidied
    sim_dir = raw_simulation_dir(group_dir)
    if sim_dir is None:
        raise FileNotFoundError(f"{group_dir}: neither {DETAILS_DIR}/ nor simulation/<challenge>/<timestamp>")
    return sim_dir


def metrics_dir(group_dir):
    return details_dir(group_dir) / "metrics"


def aggregated_metrics_csv(group_dir):
    """-> the aggregator's full table: per-scenario, per-scenario-type and final_score rows."""
    base = details_dir(group_dir)
    if (base / AGGREGATED_METRICS_CSV).is_file():
        return base / AGGREGATED_METRICS_CSV
    matches = sorted(base.glob("aggregator_metric/*_weighted_average_metrics_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one aggregator csv under {base}, found {matches}")
    return matches[0]


def scenarios_csv(group_dir):
    """-> CaRL's per-scenario score table, worst first."""
    for path in (Path(group_dir) / "scenarios.csv", details_dir(group_dir) / "aggregator_metric" / "scenarios.csv"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"no scenarios.csv under {group_dir}")


def simulation_logs(group_dir):
    """-> {scenario token: its simulation log}; nuPlan names the file <token>.msgpack.xz in both layouts."""
    log_dir = details_dir(group_dir) / "simulation_log"
    return {
        path.name[: -len(SIMULATION_LOG_SUFFIX)]: path for path in sorted(log_dir.rglob(f"*{SIMULATION_LOG_SUFFIX}"))
    }
