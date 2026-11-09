"""Tidy the output dir of scripts/kesai/12_alpasim.sh down to what is read after the run:

  results.txt    official score, km between incidents and the per-metric means (written by 12_alpasim.sh)
  run_info.txt   checkpoint, scenes, topology, git states
  obs_html/      the replay pages render_alpasim_obs_html.py renders for the selected scenes (+ index.html)
  details/       results-summary.json (the scored per-scene rows), metrics_unprocessed.parquet (per-step metrics of
                 every rollout), eval-config.yaml, wizard-config.yaml, config.json, evaluation.yaml, evaluate.log,
                 evaluation.log, services.log (every AlpaSim service container log, concatenated)
  videos/        the in-run mp4s, only with RENDER_VIDEO=1
  alpasim/wizard_0/rollouts/<clipgt>/<session>/rollout.asl  only with KEEP_ROLLOUT_LOGS all|imperfect|failed, where
                 AlpaSim's and alpasim-training's offline tools look

usage: python scripts/eval/tidy_alpasim_run.py <run_dir> [--keep_rollout_logs none] [--keep_container_info]

Everything else is deleted: the evaluated checkpoint copy (final_model.pt + config.yaml, once the source checkpoint
named in run_info.txt is confirmed to exist), nre_rc.local, per-rollout metrics.parquet and _complete markers
(concatenated in metrics_unprocessed.parquet), controller csvs, runtime worker logs (copies of the runtime lines of
evaluate.log), generated configs, docker-compose.yaml, prometheus configs, aggregate plots. evaluate.log and
services.log lose the containers' INFO lines unless --keep_container_info. A run without results-summary.json (never
aggregated: the wizard skips the aggregation when it takes a leaked SLURM_ARRAY_TASK_COUNT for an array job) keeps its
wizard tree minus the controller csvs and worker logs; `<alpasim>/.venv/bin/python -m eval.aggregation.main
--array_job_dir <wizard_0> --config_path <wizard_0>/eval-config.yaml --failed_rollouts <wizard_0>/failed_rollouts.json`
scores it, then this script finishes the job. Runs tidied into the 2026-10-12 layout (obs_replays/, logs/, alpasim/
files) are moved into this one."""

import argparse
import json
import re
import shutil
from pathlib import Path


KEEP_RULES = {
    "all": lambda row: True,
    "imperfect": lambda row: row["score"] < 1,
    "failed": lambda row: row["status"] == "fail",
    "none": lambda row: False,
}
DETAILS_DIR = "details"
OBS_HTML_DIR = "obs_html"
RUN_INFO = "run_info.txt"
EVALUATE_LOG = "evaluate.log"
SERVICES_LOG = "services.log"
ALPAGYM_LOGS = ("config.json", "evaluation.yaml", "logs/evaluation.log")
LEGACY_LOGS = ("evaluate.log", "config.json", "evaluation.yaml", "evaluation.log")  # logs/ of the 2026-10-12 layout
WIZARD_KEPT = (
    "aggregate/results-summary.json",
    "aggregate/metrics_unprocessed.parquet",
    "eval-config.yaml",
    "wizard-config.yaml",
)
RUN_SCRATCH = ("nre_rc.local",)
WORKER_LOG_GLOB = "runtime_worker_*.log"
CONTROLLER_CSV_GLOB = "controller/alpasim_controller_*.csv"
CONTAINER_INFO_LINE = re.compile(rb"^[a-z0-9_-]+-[0-9]+-[0-9]+ +\| .*( INFO:|\[INFO\])")  # docker compose prefix
SERVICE_INFO_LINE = re.compile(rb"^.*( INFO:|\[INFO\])")  # txt-logs/out-*-log.txt of enroot and pyxis runs
CHECKPOINT_LINE = re.compile(r"^checkpoint (\S+)")


def move(src, dst):
    """Rename src to dst; a src an interrupted earlier tidy already moved is skipped."""
    if not src.exists():
        return
    if dst.exists():
        raise SystemExit(f"{dst} already exists, not overwriting it with {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def remove(path):
    """-> bytes freed by deleting the file, 0 when it does not exist."""
    if not path.is_file() and not path.is_symlink():
        return 0
    size = path.lstat().st_size
    path.unlink()
    return size


def drop_info_lines(log_path, info_line):
    """-> bytes removed from log_path by dropping the lines matching info_line, AlpaSim's container INFO lines."""
    size_before = log_path.stat().st_size
    filtered_path = log_path.with_suffix(".filtered")
    with open(log_path, "rb") as source, open(filtered_path, "wb") as target:
        for line in source:
            if info_line.match(line) is None:
                target.write(line)
    filtered_path.replace(log_path)
    return size_before - log_path.stat().st_size


def remove_empty_dirs(root):
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()


def remove_all(paths):
    """-> (files deleted, bytes freed)."""
    freed_bytes = 0
    paths = list(paths)
    for path in paths:
        freed_bytes += remove(path)
    return len(paths), freed_bytes


def append_log(src, target, header):
    """Append src under a header line to target and delete it. -> bytes freed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "ab") as target_file, open(src, "rb") as src_file:
        target_file.write(f"==== {header} ====\n".encode())
        shutil.copyfileobj(src_file, target_file)
        target_file.write(b"\n")
    return remove(src)


def tidy_service_logs(txt_logs_dir, details_dir, keep_container_info):
    """txt-logs/: the runtime worker logs deleted (copies of evaluate.log's runtime lines), the service container logs
    into details/services.log. -> (files deleted, bytes freed)."""
    if not txt_logs_dir.is_dir():
        return 0, 0
    deleted_count, freed_bytes = remove_all(sorted(txt_logs_dir.glob(WORKER_LOG_GLOB)))
    for service_log in sorted(path for path in txt_logs_dir.iterdir() if path.is_file()):
        if not keep_container_info:
            freed_bytes += drop_info_lines(service_log, SERVICE_INFO_LINE)
        freed_bytes += append_log(service_log, details_dir / SERVICES_LOG, service_log.name)
    remove_empty_dirs(txt_logs_dir)
    return deleted_count, freed_bytes


def remove_checkpoint_copy(run_dir):
    """The evaluated copy of the checkpoint and its config.yaml, once run_info.txt's source checkpoint still exists
    with the same size (a downloaded run keeps them). -> bytes freed."""
    run_info = run_dir / RUN_INFO
    if not run_info.is_file():
        return 0
    match = CHECKPOINT_LINE.match(run_info.read_text())
    if match is None:
        return 0
    source, copy = Path(match.group(1)), run_dir / Path(match.group(1)).name
    if not (source.is_file() and copy.is_file() and source.stat().st_size == copy.stat().st_size):
        return 0
    return remove(copy) + remove(run_dir / "config.yaml")


def gather_run(run_dir, details_dir):
    """A fresh run's alpagym/ tree and a 2026-10-12 layout into the current one; evaluate.log into details/."""
    alpagym_dir, legacy_logs, legacy_alpasim = run_dir / "alpagym", run_dir / "logs", run_dir / "alpasim"
    move(alpagym_dir / "obs_replays", run_dir / OBS_HTML_DIR)
    for name in ALPAGYM_LOGS:
        move(alpagym_dir / name, details_dir / Path(name).name)
    move(alpagym_dir / "alpasim" / "wizard_0", legacy_alpasim / "wizard_0")
    remove_empty_dirs(alpagym_dir)
    if alpagym_dir.exists():
        leftovers = sorted(str(path.relative_to(run_dir)) for path in alpagym_dir.rglob("*") if path.is_file())
        print(f"[tidy_alpasim_run] left in place, unknown: {leftovers}")
    move(run_dir / "obs_replays", run_dir / OBS_HTML_DIR)
    move(legacy_logs / RUN_INFO, run_dir / RUN_INFO)
    for name in LEGACY_LOGS:
        move(legacy_logs / name, details_dir / name)
    move(legacy_logs / "txt-logs", legacy_alpasim / "wizard_0" / "txt-logs")
    remove_empty_dirs(legacy_logs)
    for name in WIZARD_KEPT:
        move(legacy_alpasim / Path(name).name, details_dir / Path(name).name)
    move(run_dir / EVALUATE_LOG, details_dir / EVALUATE_LOG)


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
        removed_bytes += remove(path)
        removed_count += 1
    remove_empty_dirs(wizard_dir)
    return removed_count, removed_bytes, sum(path.is_file() for path in kept_logs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="output dir of 12_alpasim.sh")
    ap.add_argument("--keep_rollout_logs", choices=tuple(KEEP_RULES), default="none")
    ap.add_argument("--keep_container_info", action="store_true", help="keep the containers' INFO lines in the logs")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    details_dir, wizard_dir = run_dir / DETAILS_DIR, run_dir / "alpasim" / "wizard_0"
    gather_run(run_dir, details_dir)
    deleted_count, freed_bytes = remove_all(run_dir / name for name in RUN_SCRATCH)
    freed_bytes += remove_checkpoint_copy(run_dir)
    evaluate_log = details_dir / EVALUATE_LOG
    if evaluate_log.is_file() and not args.keep_container_info:
        freed_bytes += drop_info_lines(evaluate_log, CONTAINER_INFO_LINE)
    controller_count, controller_bytes = remove_all(sorted(wizard_dir.glob(CONTROLLER_CSV_GLOB)))
    remove_empty_dirs(wizard_dir / "controller")
    service_count, service_bytes = tidy_service_logs(wizard_dir / "txt-logs", details_dir, args.keep_container_info)
    deleted_count += controller_count + service_count
    freed_bytes += controller_bytes + service_bytes
    summary_path = wizard_dir / "aggregate" / "results-summary.json"
    if not summary_path.is_file():
        if wizard_dir.is_dir() and any(wizard_dir.rglob("metrics.parquet")):
            print(
                f"[tidy_alpasim_run] no {summary_path}: the run was never aggregated, its wizard tree stays for "
                f"`python -m eval.aggregation.main --array_job_dir {wizard_dir}` ({deleted_count} files, "
                f"{freed_bytes / 1e9:.2f} GB removed)"
            )
        else:
            remove_empty_dirs(wizard_dir.parent)
            migrated = (
                f" ({deleted_count} files, {freed_bytes / 1e9:.2f} GB removed)" if deleted_count or freed_bytes else ""
            )
            print(f"[tidy_alpasim_run] already tidied{migrated}: {run_dir}")
        return
    rows = json.loads(summary_path.read_text())["rollouts"]
    for mp4 in sorted(wizard_dir.glob("rollouts/*/*/*.mp4")):
        move(mp4, run_dir / "videos" / mp4.name)
    for name in WIZARD_KEPT:
        move(wizard_dir / name, details_dir / Path(name).name)
    wizard_count, wizard_bytes, kept_logs = sweep_wizard(wizard_dir, rows, KEEP_RULES[args.keep_rollout_logs])
    remove_empty_dirs(wizard_dir.parent)
    deleted_count += wizard_count
    freed_bytes += wizard_bytes
    print(
        f"[tidy_alpasim_run] removed {deleted_count} files ({freed_bytes / 1e9:.2f} GB), kept {kept_logs} rollout logs "
        f"({args.keep_rollout_logs}) -> {run_dir}"
    )


if __name__ == "__main__":
    main()
