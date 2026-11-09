"""Render the interactive observation replays of a CARLA longest6 run (scripts/kesai/11_carla_longest6_v2.sh,
14_carla_longest6_v1.sh: <run_dir>/routes/route_XX[_s<seed>]/obs_html/*.replay.zlib) into <run_dir>/obs_html/, one
page per selected route run plus an index.html gallery (same navigator as the nuPlan and self-play eval replays).

usage: python scripts/eval/render_carla_obs_html.py <run_dir> [--routes <longest6.xml>] [--select failures|all]
       [--workers N]

--select failures (default) renders the route runs with a flagged leaderboard infraction or a failed status
(carla_run_layout.route_failed; min-speed infractions alone do not count), all renders every route run.
Pages are named ds<driving score>_<flags>_<town>_route<id>_s<seed>.html so the gallery lists the worst first
(flags: COL collision, RED red light, STOP stop sign, OFF outside route lanes, DEV route deviation,
BLOCK blocked, TIMEOUT route/scenario timeout); each page carries the route's leaderboard scores and
infractions as a panel, every positioned infraction linking to the replay step where the ego was nearest
(red lights: to the step the route's telemetry flagged, their message carries the light's position); from
that step on the viewer draws the ego as infringing for MARK_FRAMES, under a LEADERBOARD warning chip.
--routes (the evaluated xml) names the town of every route, else the world log does.
Retried routes leave the files of crashed attempts behind: the newest files pair with the records. On a run
tidy_carla_run.py flattened, a route's rendered page is its replay source (the page embeds it); a page whose
route was driven again under another name is replaced.
"""

import argparse
import csv
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import carla_run_layout as layout
from carla_run_layout import COLLISION_KEYS, FLAG_INFRACTIONS, REPLAY_SUFFIX, infraction_messages, route_index_rep

from pufferlib import viz
from pufferlib.ocean.cosim.carla_bridge import TOWN_OFFSETS


SELECT_MODES = ("failures", "all")
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
GALLERY_INDEX = "index.html"
PAGE_KEY_PATTERN = re.compile(r"route\d+(?:_rep\d+)?(?:_s\d+)?$")  # the route-run part of a page name


def page_key(record, seed=None):
    """-> 'route08_rep1_s1': the part of a page name that names the route run, whatever its score and flags."""
    route_idx, rep = route_index_rep(record)
    return f"route{route_idx:02d}" + (f"_rep{rep}" if rep else "") + (f"_s{seed}" if seed is not None else "")


def page_name(record, town, seed=None):
    parts = [f"ds{float(record['scores']['score_composed']):05.1f}"]
    parts += [flag for flag, keys in FLAG_INFRACTIONS if any(infraction_messages(record, key) for key in keys)]
    if town:
        parts.append(town)
    parts.append(page_key(record, seed))
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


def rendered_pages(obs_dir):
    """-> page key -> the route run's rendered page: the replay source once the run is tidied (a page embeds it)."""
    pages = {}
    for page in sorted(obs_dir.glob("*.html")):
        match = PAGE_KEY_PATTERN.search(page.stem)
        if page.name != GALLERY_INDEX and match:
            pages[match.group(0)] = page
    return pages


def collect_jobs(run_dir, towns, select):
    """-> [(record, replay source, town, world log meta, seed, telemetry csv or None)] over the selected records of
    every route run: the newest replay of a route folder, or the route run's rendered page once tidied."""
    pages = rendered_pages(Path(run_dir) / layout.OBS_HTML_DIR)
    jobs = []
    for route_run, result_path in layout.route_runs(run_dir):
        seed = layout.seed_of(route_run)
        records = layout.read_records(result_path)
        for record, stem in zip(records, layout.record_stems(result_path, len(records))):
            if select == "failures" and not layout.route_failed(record):
                continue
            replay = layout.record_file(result_path, layout.OBS_HTML_DIR, REPLAY_SUFFIX, stem)
            source = replay or pages.get(page_key(record, seed))
            if source is None:
                print(f"[{route_run}] no replay for {record.get('route_id')}, skipped")
                continue
            world_log = layout.record_file(result_path, "world_log", ".npz", stem)
            log_meta = world_log_meta(world_log) if world_log else {}
            town = towns.get(route_index_rep(record)[0]) or log_meta.get("town", "")
            telemetry = layout.record_file(result_path, "telemetry", ".csv", stem)
            jobs.append((record, source, town, log_meta, seed, telemetry))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="output dir of 11_carla_longest6_v2.sh / 14_carla_longest6_v1.sh (holds routes/)")
    ap.add_argument("--routes", help="evaluated route xml, names the town of every route")
    ap.add_argument("--select", choices=SELECT_MODES, default="failures", help="route runs that get a page")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    towns = route_towns(args.routes) if args.routes else {}
    jobs = collect_jobs(run_dir, towns, args.select)
    if not jobs:
        raise SystemExit(f"no selected route run with a replay or a rendered page under {run_dir}")
    obs_dir = run_dir / layout.OBS_HTML_DIR
    obs_dir.mkdir(exist_ok=True)
    stale_pages = rendered_pages(obs_dir)
    pages = [
        (obs_dir / page_name(record, town, seed), record, replay, town, log_meta, seed, telemetry)
        for record, replay, town, log_meta, seed, telemetry in jobs
    ]

    def render(page):
        html_path, record, replay, town, log_meta, seed, telemetry = page
        frames, frame_count = infraction_frames(record, replay, town, log_meta, telemetry)
        panel = score_panel(record, town, step_links(frames, frame_count, log_meta.get("tick_dt")), seed)
        render_page(replay, html_path, panel + viz.replay_agent_marks(ego_marks(record, frames, frame_count)))
        stale = stale_pages.get(page_key(record, seed))
        if stale is not None and stale != html_path:
            stale.unlink()  # the route run was driven again: its old page would stay in the gallery

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(render, pages))
    file_metrics = {page.name: gallery_metrics(record) for page, record, *_ in pages}
    report = run_dir / "report" / "index.html"
    links = [("Analysis report", os.path.relpath(report, obs_dir))] if report.is_file() else []
    viz.build_gallery_index(str(obs_dir), file_metrics=file_metrics, links=links)
    print(f"[render_carla_obs_html] {len(pages)} route pages rendered ({args.select}) -> {obs_dir / GALLERY_INDEX}")


if __name__ == "__main__":
    main()
