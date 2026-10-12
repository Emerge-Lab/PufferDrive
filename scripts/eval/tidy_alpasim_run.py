"""Tidy the output dir of scripts/kesai/12_alpasim.sh down to what is read after the run:

  results.txt   official score + per-metric means (written by 12_alpasim.sh)
  obs_replays/  teacher observation replays (render_alpasim_obs_html.py turns them into the gallery)
  logs/         evaluate.log, evaluation.log, evaluation.yaml, config.json, run_info.txt
  alpasim/      results-summary.json, metrics_unprocessed.parquet (every rollout's per-step metrics), eval-config.yaml,
                wizard-config.yaml
  videos/       the in-run mp4s, only with RENDER_VIDEO=1

usage: python scripts/eval/tidy_alpasim_run.py <run_dir> [--keep_rollout_logs none] [--keep_container_info]

Everything else AlpaSim writes is deleted: per-rollout metrics.parquet (concatenated in metrics_unprocessed.parquet) and
_complete markers, controller csvs, runtime worker logs (copies of evaluate.log's runtime lines), generated configs,
docker-compose.yaml, prometheus configs, aggregate plots. Kept rollout.asl stay at alpasim/wizard_0/rollouts/<clipgt>/<session>/,
where AlpaSim's and alpasim-training's offline tools look. evaluate.log loses its container INFO lines unless
--keep_container_info: AlpaSim's runtime workers and the renderer log every sim event at INFO whatever wizard.log_level says.
A run without results-summary.json (never aggregated) keeps its AlpaSim tree. Runs in the 2026-10-12 layout
(alpasim/wizard_0/ and logs/ already in place) are tidied the same way.
"""

import argparse
import json
import re
from pathlib import Path

KEEP_RULES = {
    "all": lambda row: True,
    "imperfect": lambda row: row["score"] < 1,
    "failed": lambda row: row["status"] == "fail",
    "none": lambda row: False,
}
RUN_LOGS = ("evaluate.log", "run_info.txt")
ALPAGYM_LOGS = ("config.json", "evaluation.yaml", "logs/evaluation.log")
WIZARD_KEPT = (
    "aggregate/results-summary.json",
    "aggregate/metrics_unprocessed.parquet",
    "eval-config.yaml",
    "wizard-config.yaml",
)
CONTAINER_INFO_LINE = re.compile(rb"^[a-z0-9_-]+-[0-9]+-[0-9]+ +\| .*( INFO:|\[INFO\])")


def move(src, dst_dir):
    if not src.exists():
        return
    dst = dst_dir / src.name
    if dst.exists():
        raise SystemExit(f"{dst} already exists, not overwriting it with {src}")
    dst_dir.mkdir(exist_ok=True)
    src.rename(dst)


def drop_container_info(log_path):
    """-> bytes removed from log_path by dropping the INFO lines of AlpaSim's containers."""
    size_before = log_path.stat().st_size
    filtered_path = log_path.with_suffix(".filtered")
    with open(log_path, "rb") as source, open(filtered_path, "wb") as target:
        for line in source:
            if CONTAINER_INFO_LINE.match(line) is None:
                target.write(line)
    filtered_path.replace(log_path)
    return size_before - log_path.stat().st_size


def remove_empty_dirs(root):
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()


def sweep_wizard(wizard_dir, rows, keep_rule):
    """Delete every file of the AlpaSim tree except the rollout.asl of the summary rows keep_rule keeps.
    -> (files removed, bytes removed, rollout logs kept)."""
    kept_logs = {
        wizard_dir / "rollouts" / row["clipgt_id"] / row["rollout_id"] / "rollout.asl" for row in rows if keep_rule(row)
    }
    removed_count = removed_bytes = 0
    for path in sorted(wizard_dir.rglob("*")):
        if not path.is_file() or path in kept_logs:
            continue
        removed_bytes += path.stat().st_size
        path.unlink()
        removed_count += 1
    remove_empty_dirs(wizard_dir)
    return removed_count, removed_bytes, sum(path.is_file() for path in kept_logs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="output dir of 12_alpasim.sh")
    ap.add_argument("--keep_rollout_logs", choices=tuple(KEEP_RULES), default="none")
    ap.add_argument(
        "--keep_container_info", action="store_true", help="keep the containers' INFO lines in evaluate.log"
    )
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    alpagym_dir, alpasim_dir, logs_dir = run_dir / "alpagym", run_dir / "alpasim", run_dir / "logs"

    move(alpagym_dir / "obs_replays", run_dir)
    for name in RUN_LOGS:
        move(run_dir / name, logs_dir)
    for name in ALPAGYM_LOGS:
        move(alpagym_dir / name, logs_dir)
    if (alpagym_dir / "alpasim" / "wizard_0").is_dir():
        alpasim_dir.mkdir(exist_ok=True)
        move(alpagym_dir / "alpasim" / "wizard_0", alpasim_dir)
    remove_empty_dirs(alpagym_dir)
    if alpagym_dir.exists():
        print(
            f"[tidy_alpasim_run] left in place, unknown: {sorted(str(p) for p in alpagym_dir.rglob('*') if p.is_file())}"
        )

    removed_count = removed_bytes = 0
    for stale in sorted(logs_dir.glob("runtime_worker_*.log")) + sorted(
        logs_dir.glob("controller/alpasim_controller_*.csv")
    ):
        removed_bytes += stale.stat().st_size
        stale.unlink()
        removed_count += 1
    remove_empty_dirs(logs_dir / "controller")
    evaluate_log = logs_dir / "evaluate.log"
    if evaluate_log.is_file() and not args.keep_container_info:
        removed_bytes += drop_container_info(evaluate_log)

    wizard_dir = alpasim_dir / "wizard_0"
    summary_path = wizard_dir / "aggregate" / "results-summary.json"
    if not summary_path.is_file():
        tidied = (alpasim_dir / "results-summary.json").is_file()
        print(f"[tidy_alpasim_run] {'already tidied' if tidied else f'no {summary_path}: AlpaSim tree kept as is'}")
        return
    rows = json.loads(summary_path.read_text())["rollouts"]
    for mp4 in sorted(wizard_dir.glob("rollouts/*/*/*.mp4")):
        move(mp4, run_dir / "videos")
    for name in WIZARD_KEPT:
        move(wizard_dir / name, alpasim_dir)
    wizard_count, wizard_bytes, kept_logs = sweep_wizard(wizard_dir, rows, KEEP_RULES[args.keep_rollout_logs])
    removed_count += wizard_count
    removed_bytes += wizard_bytes
    print(
        f"[tidy_alpasim_run] removed {removed_count} files ({removed_bytes / 1e9:.2f} GB), kept {kept_logs} rollout logs "
        f"({args.keep_rollout_logs}) -> {run_dir}"
    )


if __name__ == "__main__":
    main()
