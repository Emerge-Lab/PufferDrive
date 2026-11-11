"""Where the files of a CARLA longest6 run (scripts/kesai/11_carla_longest6_v2.sh, 14_carla_longest6_v1.sh) live.
While the run drives, every route run has its own folder, routes/route_<id>_s<seed>/: result.json (the leaderboard
record), evaluator_attempt<n>.log, carla_server_attempt<n>.log and, per attempt, telemetry/<stamp>.csv,
world_log/<stamp>.npz, obs_html/<stamp>.replay.zlib and carla_view/<stamp>.mp4 (retried routes leave the files of
crashed attempts behind; the newest pair with the records). scripts/eval/tidy_carla_run.py flattens the folders to
routes/<route run>[_rep<k>].result.json plus .telemetry.csv / .world_log.npz / .carla_view.mp4 of the kept route
runs; the rendered pages in obs_html/ are the only replays left."""

import json
import re
from pathlib import Path


ROUTES_DIR = "routes"
OBS_HTML_DIR = "obs_html"
RAW_RESULT_NAME = "result.json"
RESULT_SUFFIX = ".result.json"
REPLAY_SUFFIX = ".replay.zlib"
RECORD_FILE_KINDS = (("telemetry", ".csv"), ("world_log", ".npz"), ("obs_html", REPLAY_SUFFIX), ("carla_view", ".mp4"))
ROUTE_ID_PATTERN = re.compile(r"RouteScenario_(\d+)(?:_rep(\d+))?")
SEED_SUFFIX_PATTERN = re.compile(
    r"_s(\d+)$"
)  # route_08_s1: seed 1 of route 8 (the eval scripts drive one pass per seed)
COMPLETED_STATUSES = ("Completed", "Perfect")
COLLISION_KEYS = ("collisions_pedestrian", "collisions_vehicle", "collisions_layout")
FLAG_INFRACTIONS = (  # page-name flag -> leaderboard infraction keys; min-speed and emergency-vehicle infractions carry none
    ("COL", COLLISION_KEYS),
    ("RED", ("red_light",)),
    ("STOP", ("stop_infraction",)),
    ("OFF", ("outside_route_lanes",)),
    ("DEV", ("route_dev",)),
    ("BLOCK", ("vehicle_blocked",)),
    ("TIMEOUT", ("route_timeout", "scenario_timeouts")),
)


def infraction_messages(record, key):
    return record.get("infractions", {}).get(key, []) or []


def route_failed(record):
    """A route run worth its replay page and logs: a flagged leaderboard infraction or a status other than
    Completed / Perfect. Min-speed and emergency-vehicle infractions alone do not count (most v2 routes have some)."""
    if record.get("status") not in COMPLETED_STATUSES:
        return True
    return any(infraction_messages(record, key) for _, keys in FLAG_INFRACTIONS for key in keys)


def route_index_rep(record):
    """-> (route id, repetition) of a leaderboard record ('RouteScenario_12_rep0')."""
    match = ROUTE_ID_PATTERN.fullmatch(str(record.get("route_id", "")))
    if match is None:
        raise ValueError(f"unexpected route_id {record.get('route_id')!r}")
    return int(match.group(1)), int(match.group(2) or 0)


def seed_of(route_run):
    """-> the seed of a route run name ('route_08_s1' -> '1'), None for a run from before the seeds layout."""
    match = SEED_SUFFIX_PATTERN.search(route_run)
    return match.group(1) if match else None


def flat_stem(route_run, record_index):
    """-> the flat file stem of a route run's record: route_08_s1 for its first record, route_08_rep1_s1 for the next."""
    if record_index == 0:
        return route_run
    match = SEED_SUFFIX_PATTERN.search(route_run)
    if match is None:
        return f"{route_run}_rep{record_index}"
    return f"{route_run[: match.start()]}_rep{record_index}{match.group(0)}"


def read_records(result_path):
    with open(result_path) as result_file:
        return json.load(result_file)["_checkpoint"]["records"]


def is_raw(result_path):
    return Path(result_path).name == RAW_RESULT_NAME


def route_runs(run_dir):
    """-> [(route run name, its record file)]: the route folders of a driving run and the flat records of a tidied
    one, sorted by name."""
    routes_dir = Path(run_dir) / ROUTES_DIR
    runs = {path.parent.name: path for path in routes_dir.glob(f"route_*/{RAW_RESULT_NAME}")}
    runs.update({path.name[: -len(RESULT_SUFFIX)]: path for path in routes_dir.glob(f"route_*{RESULT_SUFFIX}")})
    return sorted(runs.items())


def raw_attempt_stems(route_dir):
    """-> the timestamp stems of every attempt that left a telemetry, world log, replay or chase-cam file, oldest first."""
    stems = set()
    for kind, suffix in RECORD_FILE_KINDS:
        stems.update(path.name[: -len(suffix)] for path in (Path(route_dir) / kind).glob(f"*{suffix}"))
    return sorted(stems)


def record_stems(result_path, record_count):
    """-> one file stem per record, oldest first: the newest attempt stems of a route folder (an attempt that crashed
    before writing leaves the oldest records with None), the flat stems once tidied."""
    result_path = Path(result_path)
    if not is_raw(result_path):
        return [flat_stem(result_path.name[: -len(RESULT_SUFFIX)], k) for k in range(record_count)]
    stems = raw_attempt_stems(result_path.parent)[-record_count:] if record_count else []
    return [None] * (record_count - len(stems)) + stems


def record_file(result_path, kind, suffix, stem):
    """-> the `kind` file of a record (its stem from record_stems), None when it was not written or not kept."""
    result_path = Path(result_path)
    if stem is None:
        return None
    path = (
        result_path.parent / kind / f"{stem}{suffix}"
        if is_raw(result_path)
        else result_path.parent / f"{stem}.{kind}{suffix}"
    )
    return path if path.is_file() else None
