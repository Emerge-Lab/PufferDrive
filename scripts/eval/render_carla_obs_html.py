"""Render the interactive observation replays of a CARLA longest6 run (scripts/kesai/11_carla_longest6_v2.sh:
<run_dir>/routes/route_XX[_s<seed>]/obs_html/*.replay.zlib) into <run_dir>/obs_html/, one page per route plus an
index.html gallery (same navigator as the nuPlan and self-play eval replays).

usage: python scripts/eval/render_carla_obs_html.py <run_dir> [--routes <longest6.xml>] [--workers N]

Pages are named ds<driving score>_<flags>_<town>_route<id>.html so the gallery lists the worst first
(flags: COL collision, RED red light, STOP stop sign, OFF outside route lanes, DEV route deviation,
BLOCK blocked, TIMEOUT route/scenario timeout); each page carries the route's leaderboard scores and
infractions as a panel, every positioned infraction linking to the replay step where the ego was nearest
(red lights: to the step the route's telemetry flagged, their message carries the light's position); from
that step on the viewer draws the ego as infringing for MARK_FRAMES, under a LEADERBOARD warning chip.
--routes (the evaluated xml) names the town of every route, else the world log does.
Retried routes leave the files of crashed attempts behind: the newest files pair with the records.
"""

import argparse
import csv
import glob
import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from pufferlib import viz
from pufferlib.ocean.cosim.carla_bridge import TOWN_OFFSETS


COLLISION_KEYS = ("collisions_pedestrian", "collisions_vehicle", "collisions_layout")
FLAG_INFRACTIONS = (
    ("COL", COLLISION_KEYS),
    ("RED", ("red_light",)),
    ("STOP", ("stop_infraction",)),
    ("OFF", ("outside_route_lanes",)),
    ("DEV", ("route_dev",)),
    ("BLOCK", ("vehicle_blocked",)),
    ("TIMEOUT", ("route_timeout", "scenario_timeouts")),
)
PANEL_INFRACTIONS = (
    ("collisions_pedestrian", "pedestrian collisions"),
    ("collisions_vehicle", "vehicle collisions"),
    ("collisions_layout", "layout collisions"),
    ("red_light", "red lights"),
    ("stop_infraction", "stop signs"),
    ("outside_route_lanes", "outside route lanes"),
    ("route_dev", "route deviations"),
    ("vehicle_blocked", "blocked"),
    ("route_timeout", "route timeouts"),
    ("scenario_timeouts", "scenario timeouts"),
    ("min_speed_infractions", "min speed"),
    ("yield_emergency_vehicle_infractions", "emergency vehicles"),
)
MESSAGE_MAX_CHARS = 120
POSITION_PATTERN = re.compile(r"x=([-\d.]+), y=([-\d.]+)")
NEAREST_STEP_MAX_M = 5.0
RED_LIGHT_FLAG_COLUMN = "carla_infr_red"  # telemetry: 1 on the step the simulator-side red light test fired
EGO_AGENT_IDX = 0
MARK_FRAMES = 40  # 2 s at the leaderboards' 20 ticks per second
MARK_NAMES = {
    "collisions_pedestrian": "LEADERBOARD COLLISION",
    "collisions_vehicle": "LEADERBOARD COLLISION",
    "collisions_layout": "LEADERBOARD COLLISION",
    "red_light": "LEADERBOARD RED LIGHT",
    "stop_infraction": "LEADERBOARD STOP SIGN",
    "route_dev": "LEADERBOARD ROUTE DEVIATION",
    "vehicle_blocked": "LEADERBOARD BLOCKED",
}
REPLAY_SUFFIX = ".replay.zlib"
ROUTE_ID_PATTERN = re.compile(r"RouteScenario_(\d+)(?:_rep(\d+))?")
SEED_DIR_PATTERN = re.compile(r"_s(\d+)$")  # route_08_s1: seed 1 of route 8 (the eval scripts drive one pass per seed)


def infraction_messages(record, key):
    return record.get("infractions", {}).get(key, []) or []


def route_index_rep(record):
    """-> (route id, repetition) of a leaderboard record ('RouteScenario_12_rep0')."""
    match = ROUTE_ID_PATTERN.fullmatch(str(record.get("route_id", "")))
    if match is None:
        raise ValueError(f"unexpected route_id {record.get('route_id')!r}")
    return int(match.group(1)), int(match.group(2) or 0)


def page_name(record, town, seed=None):
    route_idx, rep = route_index_rep(record)
    parts = [f"ds{float(record['scores']['score_composed']):05.1f}"]
    parts += [flag for flag, keys in FLAG_INFRACTIONS if any(infraction_messages(record, key) for key in keys)]
    if town:
        parts.append(town)
    parts.append(f"route{route_idx:02d}" + (f"_rep{rep}" if rep else "") + (f"_s{seed}" if seed is not None else ""))
    return "_".join(parts) + ".html"


def score_panel(record, town, steps, seed=None):
    """Fixed panel injected into the viewer: leaderboard scores, every infraction category, the messages + steps."""
    scores, meta = record["scores"], record.get("meta", {})
    route_idx, rep = route_index_rep(record)
    lines = []
    for key, label in PANEL_INFRACTIONS:
        messages = infraction_messages(record, key)
        color = "#ff6b6b" if messages else "#7ed491"
        mark = "&#10007;" if messages else "&#10003;"
        lines.append(
            f'<div style="display:flex;justify-content:space-between;gap:16px"><span style="color:{color}">{mark} {label}</span>'
            f'<span style="color:{color}">{len(messages)}</span></div>'
        )
        for message in messages:
            step_html = f" &middot; {steps[message]}" if message in steps else ""
            lines.append(
                f'<div style="color:#ff6b6b;padding-left:14px">&#8627; {html.escape(message[:MESSAGE_MAX_CHARS])}'
                f"{step_html}</div>"
            )
    if meta:
        lines.append(
            f'<div style="color:#7f8ba0">route {float(meta.get("route_length", 0)):.0f} m &middot; '
            f"game {float(meta.get('duration_game', 0)):.0f} s &middot; wall {float(meta.get('duration_system', 0)):.0f} s</div>"
        )
    title = (
        f"{town + ' ' if town else ''}route {route_idx}"
        + (f" rep {rep}" if rep else "")
        + (f" seed {seed}" if seed is not None else "")
    )
    return (
        '<details open style="position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:1000;'
        "background:rgba(13,20,32,.93);color:#c4cddc;font:12px/1.5 ui-monospace,monospace;padding:6px 12px;"
        'border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.4);max-width:640px">'
        f'<summary style="cursor:pointer;font-weight:600">driving score {float(scores["score_composed"]):.1f} &middot; '
        f"RC {float(scores['score_route']):.0f}% &middot; IP {float(scores['score_penalty']):.3f} &middot; "
        f"{html.escape(title)} &middot; {html.escape(str(record.get('status', '')))}</summary>"
        + "".join(lines)
        + "</details>"
    )


def red_light_steps(telemetry_path):
    """Steps on which the route's telemetry flagged a run red light (row k of the CSV = replay frame k)."""
    with open(telemetry_path, newline="") as telemetry_file:
        rows = csv.DictReader(telemetry_file)
        return [step for step, row in enumerate(rows) if float(row[RED_LIGHT_FLAG_COLUMN]) > 0.0]


def infraction_frames(record, replay_path, town, log_meta, telemetry_path=None):
    """-> (message -> replay frame of every infraction that can be placed, the replay's frame count): red lights at
    the frame the telemetry flagged, the other positioned ones where the ego was nearest (CARLA -> bin offset: the
    world log's, else the town's). A frame at or past the count: the infraction happened after the replay ended."""
    _, chunks = viz.read_replay_zlib(replay_path, chunk_names=("agent_f32",))
    ego_xy = chunks["agent_f32"][:, 0, :2]
    frame_count = len(ego_xy)
    frames = {}
    red_messages = infraction_messages(record, "red_light")
    flagged_steps = red_light_steps(telemetry_path) if telemetry_path and red_messages else []
    if len(flagged_steps) == len(red_messages):
        frames.update(zip(red_messages, flagged_steps))
    offset = log_meta.get("offset") or TOWN_OFFSETS.get(town)
    nearest_keys = [key for key, _ in PANEL_INFRACTIONS if key != "red_light" and offset is not None]
    for message in [message for key in nearest_keys for message in infraction_messages(record, key)]:
        match = POSITION_PATTERN.search(message)
        if match is None:
            continue
        bin_x, bin_y = float(match.group(1)) + offset[0], -float(match.group(2)) + offset[1]
        distance_m = np.hypot(ego_xy[:, 0] - bin_x, ego_xy[:, 1] - bin_y)
        nearest_frame = int(np.argmin(distance_m))
        # a place the replay's ego never came near was reached after its last frame
        frames[message] = nearest_frame if distance_m[nearest_frame] <= NEAREST_STEP_MAX_M else frame_count
    return frames, frame_count


def step_links(frames, frame_count, tick_dt):
    """message -> the panel's seek link to the infraction's frame, or the note that the replay ended before it."""
    links = {}
    for message, frame in frames.items():
        seconds = None if tick_dt is None else frame * tick_dt
        links[message] = viz.replay_step_link(frame, seconds) if frame < frame_count else "beyond the replay"
    return links


def ego_marks(record, frames, frame_count):
    """-> viz.replay_agent_marks rows: the ego counts as infringing for MARK_FRAMES from each placed infraction on."""
    keyed_messages = [(key, message) for key in MARK_NAMES for message in infraction_messages(record, key)]
    return [
        (EGO_AGENT_IDX, frames[message], MARK_FRAMES, MARK_NAMES[key])
        for key, message in keyed_messages
        if frames.get(message, frame_count) < frame_count
    ]


def render_page(zlib_path, html_path, panel):
    viz.render_interactive_replay_zlib(str(zlib_path), str(html_path))
    page = Path(html_path).read_text()
    Path(html_path).write_text(page.replace("<body>", "<body>" + panel, 1))


def gallery_metrics(record):
    """Infraction counts under the names the gallery filters on (> 0 = flagged)."""
    return {
        "driving_score": float(record["scores"]["score_composed"]),
        "collision_rate": float(sum(len(infraction_messages(record, key)) for key in COLLISION_KEYS)),
        "red_light_violation_rate": float(len(infraction_messages(record, "red_light"))),
        "offroad_rate": float(len(infraction_messages(record, "outside_route_lanes"))),
    }


def route_towns(routes_xml):
    return {
        int(route.attrib["id"]): route.attrib.get("town", "") for route in ET.parse(routes_xml).getroot().iter("route")
    }


def world_log_meta(npz_path):
    return json.loads(str(np.load(npz_path)["meta"]))


def newest_files(route_dir, subdir, suffix, count):
    """The `count` newest (timestamp-named) files of <route_dir>/<subdir>."""
    files = sorted(glob.glob(str(route_dir / subdir / f"*{suffix}")))
    return files[max(0, len(files) - count) :]


def collect_jobs(run_dir, towns):
    """-> [(record, replay zlib path, town, world log meta, seed or None, telemetry csv path or None)] over
    routes/route_*, the newest replay per record."""
    jobs = []
    for route_dir in sorted(Path(run_dir, "routes").glob("route_*")):
        seed_match = SEED_DIR_PATTERN.search(route_dir.name)
        seed = seed_match.group(1) if seed_match else None
        result_path = route_dir / "result.json"
        if not result_path.is_file():
            print(f"[{route_dir.name}] no result.json, skipped")
            continue
        with open(result_path) as result_file:
            records = json.load(result_file)["_checkpoint"]["records"]
        replays = newest_files(route_dir, "obs_html", REPLAY_SUFFIX, len(records))
        if len(replays) != len(records):
            print(f"[{route_dir.name}] {len(records)} route records but {len(replays)} replays; pairing the newest")
        if not replays:
            continue
        records = records[-len(replays) :]
        world_logs = newest_files(route_dir, "world_log", ".npz", len(replays))
        for k, (record, replay) in enumerate(zip(records, replays)):
            log_meta = world_log_meta(world_logs[k]) if k < len(world_logs) else {}
            town = towns.get(route_index_rep(record)[0]) or log_meta.get("town", "")
            telemetry = route_dir / "telemetry" / (Path(replay).name[: -len(REPLAY_SUFFIX)] + ".csv")
            jobs.append((record, replay, town, log_meta, seed, telemetry if telemetry.is_file() else None))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="output dir of 11_carla_longest6_v2.sh (holds routes/route_*/)")
    ap.add_argument("--routes", help="evaluated route xml, names the town of every route")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    towns = route_towns(args.routes) if args.routes else {}
    jobs = collect_jobs(run_dir, towns)
    if not jobs:
        raise SystemExit(f"no routes/route_*/obs_html/*.replay.zlib with a result.json under {run_dir}")
    obs_dir = run_dir / "obs_html"
    obs_dir.mkdir(exist_ok=True)
    pages = [
        (obs_dir / page_name(record, town, seed), record, replay, town, log_meta, seed, telemetry)
        for record, replay, town, log_meta, seed, telemetry in jobs
    ]

    def render(page):
        html_path, record, replay, town, log_meta, seed, telemetry = page
        frames, frame_count = infraction_frames(record, replay, town, log_meta, telemetry)
        panel = score_panel(record, town, step_links(frames, frame_count, log_meta.get("tick_dt")), seed)
        render_page(replay, html_path, panel + viz.replay_agent_marks(ego_marks(record, frames, frame_count)))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(render, pages))
    file_metrics = {page.name: gallery_metrics(record) for page, record, *_ in pages}
    report = run_dir / "report" / "index.html"
    links = [("Analysis report", os.path.relpath(report, obs_dir))] if report.is_file() else []
    viz.build_gallery_index(str(obs_dir), file_metrics=file_metrics, links=links)
    print(f"[render_carla_obs_html] {len(pages)} route pages rendered -> {obs_dir / 'index.html'}")


if __name__ == "__main__":
    main()
