"""Tidy a CARLA longest6 run down to results.csv, run_info.txt, obs_html/, flat routes/ and details/."""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import carla_run_layout as layout
from tidy_alpasim_run import append_log, move, remove, remove_all, remove_empty_dirs


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
COMPAT_LIB_DIR = "compat_lib"


def tidy_route_run(route_dir, routes_dir, details_dir, keep):
    """Flatten one route folder into routes/ and details/evaluators.log. -> (files deleted, bytes freed, files kept)."""
    result_path = route_dir / layout.RAW_RESULT_NAME
    records = layout.read_records(result_path) if result_path.is_file() else []
    stems = layout.record_stems(result_path, len(records))
    kept_stems = {
        stem: k
        for k, (stem, record) in enumerate(zip(stems, records))
        if stem and (keep == "all" or (keep == "failures" and layout.route_failed(record)))
    }
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
    server_log_count, server_log_bytes = remove_all(sorted(route_dir.glob(SERVER_LOG_GLOB)))
    if records:
        move(result_path, routes_dir / f"{route_dir.name}{layout.RESULT_SUFFIX}")
        kept_count += 1
    else:
        freed_bytes += remove(result_path)  # the leaderboard's empty stub of a route whose every attempt crashed
    remove_empty_dirs(route_dir)
    return deleted_count + server_log_count, freed_bytes + server_log_bytes, kept_count


def tidy_run_files(run_dir, details_dir):
    """results.csv up, logs and timing into details/, aggregate/ and run scratch deleted. -> (deleted, freed bytes)."""
    aggregate_dir, compat_lib_dir = run_dir / AGGREGATE_DIR, run_dir / COMPAT_LIB_DIR
    freed_bytes = 0
    if (aggregate_dir / RESULTS_CSV).is_file():
        freed_bytes += remove(run_dir / RESULTS_CSV)  # a completion re-parsed the run
        move(aggregate_dir / RESULTS_CSV, run_dir / RESULTS_CSV)
    deleted_count, aggregate_bytes = remove_all(sorted(path for path in aggregate_dir.rglob("*") if path.is_file()))
    remove_empty_dirs(aggregate_dir)
    for path in sorted(run_dir.glob(WORKER_LOG_GLOB), key=lambda path: int(path.stem[len("worker_w") :])):
        freed_bytes += append_log(path, details_dir / WORKERS_LOG, path.name)
    timing = run_dir / TIMING_CSV
    if timing.is_file() and (details_dir / TIMING_CSV).is_file():
        with open(timing) as timing_file, open(details_dir / TIMING_CSV, "a") as target_file:
            target_file.writelines(timing_file.readlines()[1:])  # a completion's rows join the run's table
        freed_bytes += remove(timing)
    else:
        move(timing, details_dir / TIMING_CSV)
    freed_bytes += sum(remove(run_dir / name) for name in RUN_SCRATCH)
    compat_lib_count, compat_lib_bytes = remove_all(sorted(compat_lib_dir.iterdir()) if compat_lib_dir.is_dir() else [])
    remove_empty_dirs(compat_lib_dir)
    return deleted_count + compat_lib_count, freed_bytes + aggregate_bytes + compat_lib_bytes


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
