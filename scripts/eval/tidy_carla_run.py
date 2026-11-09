"""Tidy a CARLA longest6 run (the $OUT folder of scripts/kesai/11_carla_longest6_v2.sh or 14_carla_longest6_v1.sh,
after its gallery was rendered) down to what is read after the run:

  results.csv     the result parser's summary and per-route table (moved up from aggregate/)
  run_info.txt    the run's settings and speed line (COMPLETE_RUN reads it)
  obs_html/       the replay pages render_carla_obs_html.py rendered before this script ran
  routes/         flat: route_<id>_s<seed>.result.json of every route run; .telemetry.csv, .world_log.npz and
                  .carla_view.mp4 of the kept route runs (route_<id>_rep<k>_s<seed>.* for further repetitions)
  details/        timing.csv, workers.log (every worker log), evaluators.log (every evaluator log)

usage: python scripts/eval/tidy_carla_run.py <run_dir> [--keep failures]

--keep selects the route runs whose telemetry, world log and chase cam survive: failures (a flagged leaderboard
infraction or a failed status, the gallery's default selection), all, none. Everything else is deleted: the
aggregate/ copies of the records, every .replay.zlib (a rendered page embeds its replay), the files of crashed
earlier attempts, the CARLA server logs, server_pids, compat_lib/ and the queue. report/ and pre_completion_*/
stay. A folder tidied before is left alone; a completed run (COMPLETE_RUN) is tidied again for its new routes."""

import argparse
import shutil
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import carla_run_layout as layout
from tidy_alpasim_run import remove_empty_dirs


KEEP_MODES = ("failures", "all", "none")
DETAILS_DIR = "details"
AGGREGATE_DIR = "aggregate"
RESULTS_CSV = "results.csv"
TIMING_CSV = "timing.csv"
WORKERS_LOG = "workers.log"
EVALUATORS_LOG = "evaluators.log"
WORKER_LOG_GLOB = "worker_w*.log"
EVALUATOR_LOG_GLOB = "evaluator_attempt*.log"
SERVER_LOG_GLOB = "carla_server_attempt*.log"
RUN_SCRATCH = ("server_pids", "queue", "queue.lock")
RUN_SCRATCH_DIRS = ("compat_lib",)


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


def append_log(src, target, header):
    """Append src under a header line to target (details/*.log) and delete it. -> bytes freed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "ab") as target_file, open(src, "rb") as src_file:
        target_file.write(f"==== {header} ====\n".encode())
        shutil.copyfileobj(src_file, target_file)
        target_file.write(b"\n")
    return remove(src)


def worker_index(path):
    return int(path.stem[len("worker_w") :])


def kept_record(record, keep):
    if keep == "all":
        return True
    if keep == "none":
        return False
    return layout.route_failed(record)


def tidy_route_run(route_dir, routes_dir, details_dir, keep):
    """Flatten one route folder: the record and the kept records' files up into routes/, the evaluator logs into
    details/evaluators.log, the rest deleted. -> (files deleted, bytes freed, files kept)."""
    result_path = route_dir / layout.RAW_RESULT_NAME
    records = layout.read_records(result_path) if result_path.is_file() else []
    stems = layout.record_stems(result_path, len(records))
    kept_stems = {stem: k for k, (stem, record) in enumerate(zip(stems, records)) if stem and kept_record(record, keep)}
    deleted_count = freed_bytes = kept_count = 0
    for kind, suffix in layout.RECORD_FILE_KINDS:
        for path in sorted((route_dir / kind).glob(f"*{suffix}")):
            stem = path.name[: -len(suffix)]
            if kind != layout.OBS_HTML_DIR and stem in kept_stems:
                move(path, routes_dir / f"{layout.flat_stem(route_dir.name, kept_stems[stem])}.{kind}{suffix}")
                kept_count += 1
                continue
            freed_bytes += remove(path)
            deleted_count += 1
    for path in sorted(route_dir.glob(EVALUATOR_LOG_GLOB)):
        freed_bytes += append_log(path, details_dir / EVALUATORS_LOG, f"{route_dir.name} {path.name}")
    for path in sorted(route_dir.glob(SERVER_LOG_GLOB)):
        freed_bytes += remove(path)
        deleted_count += 1
    if records:
        move(result_path, routes_dir / f"{route_dir.name}{layout.RESULT_SUFFIX}")
        kept_count += 1
    else:
        freed_bytes += remove(result_path)  # the leaderboard's empty stub of a route whose every attempt crashed
    remove_empty_dirs(route_dir)
    return deleted_count, freed_bytes, kept_count


def tidy_run_files(run_dir, details_dir):
    """results.csv up from aggregate/, the aggregate copies, worker logs, timing and run scratch into details/ or
    away. -> (files deleted, bytes freed)."""
    deleted_count = freed_bytes = 0
    aggregate_dir = run_dir / AGGREGATE_DIR
    if (aggregate_dir / RESULTS_CSV).is_file():
        freed_bytes += remove(run_dir / RESULTS_CSV)  # a completion re-parsed the run
        move(aggregate_dir / RESULTS_CSV, run_dir / RESULTS_CSV)
    if aggregate_dir.is_dir():
        for path in sorted(aggregate_dir.rglob("*")):
            if path.is_file():
                freed_bytes += remove(path)
                deleted_count += 1
        remove_empty_dirs(aggregate_dir)
    for path in sorted(run_dir.glob(WORKER_LOG_GLOB), key=worker_index):
        freed_bytes += append_log(path, details_dir / WORKERS_LOG, path.name)
    timing = run_dir / TIMING_CSV
    if timing.is_file() and (details_dir / TIMING_CSV).is_file():
        with open(timing) as timing_file, open(details_dir / TIMING_CSV, "a") as target_file:
            target_file.writelines(timing_file.readlines()[1:])  # a completion's rows join the run's table
        freed_bytes += remove(timing)
    else:
        move(timing, details_dir / TIMING_CSV)
    for name in RUN_SCRATCH:
        freed_bytes += remove(run_dir / name)
    for name in RUN_SCRATCH_DIRS:
        if (run_dir / name).is_dir():
            for path in sorted((run_dir / name).iterdir()):
                freed_bytes += remove(path)
                deleted_count += 1
            remove_empty_dirs(run_dir / name)
    return deleted_count, freed_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--keep", choices=KEEP_MODES, default="failures", help="route runs whose logs survive")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    routes_dir = run_dir / layout.ROUTES_DIR
    details_dir = run_dir / DETAILS_DIR
    if not routes_dir.is_dir():
        raise SystemExit(f"no {layout.ROUTES_DIR}/ under {run_dir}")
    route_dirs = sorted(path for path in routes_dir.glob("route_*") if path.is_dir())
    if not route_dirs and not (run_dir / AGGREGATE_DIR).exists() and not list(run_dir.glob(WORKER_LOG_GLOB)):
        print(f"[tidy_carla_run] already tidied: {run_dir}")
        return
    deleted_count = freed_bytes = kept_count = 0
    for route_dir in route_dirs:
        route_deleted, route_freed, route_kept = tidy_route_run(route_dir, routes_dir, details_dir, args.keep)
        deleted_count += route_deleted
        freed_bytes += route_freed
        kept_count += route_kept
    run_deleted, run_freed = tidy_run_files(run_dir, details_dir)
    deleted_count += run_deleted
    freed_bytes += run_freed
    leftovers = sorted(path.name for path in route_dirs if path.exists())
    if leftovers:
        print(f"[tidy_carla_run] route folders left in place, unknown files: {leftovers}")
    print(
        f"[tidy_carla_run] removed {deleted_count} files ({freed_bytes / 1e9:.2f} GB freed), kept {kept_count} "
        f"route files ({args.keep}) over {len(route_dirs)} route folders -> {run_dir}"
    )


if __name__ == "__main__":
    main()
