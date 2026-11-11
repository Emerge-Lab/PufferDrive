"""Tidy a CARLA self-play eval (scripts/parallel_eval.py, or one `puffer eval` run) down to what is read after it:

  evaluation_summary.json   the metric averages over every scenario (eval_wandb_report.py reads them)
  obs_html/                 the rendered replay pages of every shard's failures + one index.html
  details/                  episode_metrics.csv (one row per scenario), failure_episodes.csv (the re-simulated
                            failure episodes behind the pages, with their shard), selected_failures.csv (the rows
                            the render filter picked, with their shard), resolved_benchmark.yaml, shards.log

usage: python scripts/eval/tidy_selfplay_eval.py <aggregate dir>

<aggregate dir> is <run>/eval/carla_<name>/<stamp> as parallel_eval.py leaves it, its shards next to it as
<stamp>_shard<NN>/ (resolved_benchmark.yaml, episode_metrics.csv, evaluation_summary.json, failures/{...,
rendered_replays/}); a single-process `puffer eval` dir holds the same files itself. The shard folders are deleted:
their episode_metrics.csv rows are the aggregate's, their summaries are per-shard means, the per-shard gallery
indexes make way for one. resolved_benchmark.yaml is the first shard's (the others differ in their scenario window
only). A .replay.zlib kept by eval.keep_zlib_replays is deleted once its page exists. An already tidied dir is
left alone."""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tidy_alpasim_run import remove_empty_dirs

from pufferlib import viz


OBS_HTML_DIR = "obs_html"
DETAILS_DIR = "details"
SUMMARY_JSON = "evaluation_summary.json"
EPISODE_METRICS_CSV = "episode_metrics.csv"
FAILURE_EPISODES_CSV = "failure_episodes.csv"
SELECTED_FAILURES_CSV = "selected_failures.csv"
RESOLVED_BENCHMARK_YAML = "resolved_benchmark.yaml"
SHARDS_LOG = "shards.log"
SHARD_LOG_GLOB = "shard*.log"
FAILURES_DIR = "failures"
RENDERED_REPLAYS_DIR = "rendered_replays"
ZLIB_REPLAYS_DIR = "replays_zlib"
REPLAY_SUFFIX = ".replay.zlib"
GALLERY_INDEX = "index.html"
SHARD_COLUMN = "shard"


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
    if not path.is_file():
        return 0
    size = path.stat().st_size
    path.unlink()
    return size


def append_log(src, target, header):
    """Append src under a header line to target and delete it. -> bytes freed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "ab") as target_file, open(src, "rb") as src_file:
        target_file.write(f"==== {header} ====\n".encode())
        shutil.copyfileobj(src_file, target_file)
        target_file.write(b"\n")
    return remove(src)


def shard_dirs(run_dir):
    """-> the <stamp>_shard<NN> folders next to a parallel_eval.py aggregate dir, then the dir itself when it holds
    a run's own failures/ or resolved_benchmark.yaml (a single-process `puffer eval`)."""
    shards = sorted(path for path in run_dir.parent.glob(f"{run_dir.name}_shard*") if path.is_dir())
    if (run_dir / FAILURES_DIR).is_dir() or (run_dir / RESOLVED_BENCHMARK_YAML).is_file():
        shards.append(run_dir)
    return shards


def page_name(replay_path):
    """-> the page rendered from a replay: <stem>.html for <stem>.replay.zlib."""
    name = Path(str(replay_path)).name
    return name[: -len(REPLAY_SUFFIX)] + ".html" if name.endswith(REPLAY_SUFFIX) else name


def read_csv_with_shard(path, shard):
    frame = pd.read_csv(path)
    frame.insert(0, SHARD_COLUMN, shard)
    return frame


def tidy_shard(shard_dir, run_dir, details_dir, obs_dir, keep_aggregate_metrics):
    """Pages into obs_html/, the failure tables and the benchmark config into details/, the rest deleted.
    -> (failure episode frame or None, selected failures frame or None, files deleted, bytes freed)."""
    shard = shard_dir.name[len(run_dir.name) + 1 :] if shard_dir != run_dir else "single"
    failures_dir = shard_dir / FAILURES_DIR
    deleted_count = freed_bytes = 0
    for page in sorted((failures_dir / RENDERED_REPLAYS_DIR).glob("*.html")):
        if page.name == GALLERY_INDEX:
            freed_bytes += remove(page)
            deleted_count += 1
            continue
        move(page, obs_dir / page.name)
    for replay in sorted((failures_dir / ZLIB_REPLAYS_DIR).glob(f"*{REPLAY_SUFFIX}")):
        if (obs_dir / page_name(replay)).is_file():
            freed_bytes += remove(replay)
            deleted_count += 1
        else:
            move(replay, obs_dir / replay.name)
    failure_frame = selected_frame = None
    if (failures_dir / EPISODE_METRICS_CSV).is_file():
        failure_frame = read_csv_with_shard(failures_dir / EPISODE_METRICS_CSV, shard)
        freed_bytes += remove(failures_dir / EPISODE_METRICS_CSV)
        deleted_count += 1
    if (failures_dir / SELECTED_FAILURES_CSV).is_file():
        selected_frame = read_csv_with_shard(failures_dir / SELECTED_FAILURES_CSV, shard)
        freed_bytes += remove(failures_dir / SELECTED_FAILURES_CSV)
        deleted_count += 1
    freed_bytes += remove(failures_dir / SUMMARY_JSON)
    if (details_dir / RESOLVED_BENCHMARK_YAML).is_file():
        freed_bytes += remove(shard_dir / RESOLVED_BENCHMARK_YAML)
    else:
        move(shard_dir / RESOLVED_BENCHMARK_YAML, details_dir / RESOLVED_BENCHMARK_YAML)
    if shard_dir != run_dir or not keep_aggregate_metrics:
        freed_bytes += remove(shard_dir / EPISODE_METRICS_CSV)
        deleted_count += 1
    if shard_dir != run_dir:
        freed_bytes += remove(shard_dir / SUMMARY_JSON)
        deleted_count += 1
    remove_empty_dirs(failures_dir)
    if shard_dir != run_dir:
        remove_empty_dirs(shard_dir)
    return failure_frame, selected_frame, deleted_count, freed_bytes


def gallery_metrics(failure_frame):
    """page name -> its episode's finite numeric metrics, the gallery's filter inputs (as the shards' renderer)."""
    if failure_frame is None or "replay_path" not in failure_frame.columns:
        return {}
    metrics = {}
    for _, row in failure_frame.iterrows():
        values = {}
        for column, value in row.items():
            if column in (SHARD_COLUMN, "replay_path") or not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            if np.isfinite(value):
                values[column] = float(value)
        metrics[page_name(row["replay_path"])] = values
    return metrics


def append_frames(frames, target):
    """Concatenate the shards' tables under target (appending to a table an earlier tidy wrote)."""
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    if target.is_file():
        merged = pd.concat([pd.read_csv(target), merged], ignore_index=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(target, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="parallel_eval.py aggregate dir (its shards next to it), or a `puffer eval` dir")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / SUMMARY_JSON).is_file():
        raise SystemExit(f"no {SUMMARY_JSON} under {run_dir}: not an evaluated run")
    details_dir, obs_dir = run_dir / DETAILS_DIR, run_dir / OBS_HTML_DIR
    shards = shard_dirs(run_dir)
    shard_logs = sorted(run_dir.glob(SHARD_LOG_GLOB))
    if not shards and not shard_logs and not (run_dir / EPISODE_METRICS_CSV).is_file():
        print(f"[tidy_selfplay_eval] already tidied: {run_dir}")
        return
    move(run_dir / EPISODE_METRICS_CSV, details_dir / EPISODE_METRICS_CSV)
    deleted_count = freed_bytes = 0
    for log in shard_logs:
        freed_bytes += append_log(log, details_dir / SHARDS_LOG, log.name)
    failure_frames, selected_frames = [], []
    for shard_dir in shards:
        failure_frame, selected_frame, shard_deleted, shard_freed = tidy_shard(
            shard_dir, run_dir, details_dir, obs_dir, keep_aggregate_metrics=True
        )
        failure_frames.append(failure_frame)
        selected_frames.append(selected_frame)
        deleted_count += shard_deleted
        freed_bytes += shard_freed
    append_frames(failure_frames, details_dir / FAILURE_EPISODES_CSV)
    append_frames(selected_frames, details_dir / SELECTED_FAILURES_CSV)
    pages = sorted(path for path in obs_dir.glob("*.html") if path.name != GALLERY_INDEX) if obs_dir.is_dir() else []
    if pages:
        file_metrics = (
            gallery_metrics(pd.read_csv(details_dir / FAILURE_EPISODES_CSV))
            if (details_dir / FAILURE_EPISODES_CSV).is_file()
            else {}
        )
        viz.build_gallery_index(str(obs_dir), file_metrics=file_metrics)
    leftovers = sorted(
        str(path.relative_to(run_dir.parent))
        for shard in shards
        for path in shard.rglob("*")
        if path.is_file() and shard != run_dir
    )
    if leftovers:
        print(f"[tidy_selfplay_eval] left in place, unknown: {leftovers}")
    print(
        f"[tidy_selfplay_eval] {len(shards)} shard folders folded into {run_dir}: {len(pages)} replay pages, "
        f"{deleted_count} files removed ({freed_bytes / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
