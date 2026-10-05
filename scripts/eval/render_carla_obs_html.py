"""Render the interactive observation replays of a CARLA longest6 run (scripts/kesai/11_carla_longest6.sh:
<run_dir>/routes/route_XX/obs_html/*.replay.zlib) into <run_dir>/obs_html/, one page per route plus an
index.html gallery (same navigator as the nuPlan and self-play eval replays).

usage: python scripts/eval/render_carla_obs_html.py <run_dir> [--routes <longest6.xml>] [--workers N]

Pages are named ds<driving score>_<flags>_<town>_route<id>.html so the gallery lists the worst first
(flags: COL collision, RED red light, STOP stop sign, OFF outside route lanes, DEV route deviation,
BLOCK blocked, TIMEOUT route/scenario timeout); each page carries the route's leaderboard scores and
infractions as a panel. --routes (the evaluated xml) names the town of every route, else the world log does.
Retried routes leave the files of crashed attempts behind: the newest files pair with the records.
"""

import argparse
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
ROUTE_ID_PATTERN = re.compile(r"RouteScenario_(\d+)(?:_rep(\d+))?")


def infraction_messages(record, key):
    return record.get("infractions", {}).get(key, []) or []


def route_index_rep(record):
    """-> (route id, repetition) of a leaderboard record ('RouteScenario_12_rep0')."""
    match = ROUTE_ID_PATTERN.fullmatch(str(record.get("route_id", "")))
    if match is None:
        raise ValueError(f"unexpected route_id {record.get('route_id')!r}")
    return int(match.group(1)), int(match.group(2) or 0)


def page_name(record, town):
    route_idx, rep = route_index_rep(record)
    parts = [f"ds{float(record['scores']['score_composed']):05.1f}"]
    parts += [flag for flag, keys in FLAG_INFRACTIONS if any(infraction_messages(record, key) for key in keys)]
    if town:
        parts.append(town)
    parts.append(f"route{route_idx:02d}" + (f"_rep{rep}" if rep else ""))
    return "_".join(parts) + ".html"


def score_panel(record, town):
    """Fixed panel injected into the viewer: leaderboard scores, every infraction category, the messages."""
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
            lines.append(
                f'<div style="color:#ff6b6b;padding-left:14px">&#8627; {html.escape(message[:MESSAGE_MAX_CHARS])}</div>'
            )
    if meta:
        lines.append(
            f'<div style="color:#7f8ba0">route {float(meta.get("route_length", 0)):.0f} m &middot; '
            f"game {float(meta.get('duration_game', 0)):.0f} s &middot; wall {float(meta.get('duration_system', 0)):.0f} s</div>"
        )
    title = f"{town + ' ' if town else ''}route {route_idx}" + (f" rep {rep}" if rep else "")
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


def world_log_town(npz_path):
    return json.loads(str(np.load(npz_path)["meta"])).get("town", "")


def newest_files(route_dir, subdir, suffix, count):
    """The `count` newest (timestamp-named) files of <route_dir>/<subdir>."""
    files = sorted(glob.glob(str(route_dir / subdir / f"*{suffix}")))
    return files[max(0, len(files) - count) :]


def collect_jobs(run_dir, towns):
    """-> [(record, replay zlib path, town)] over routes/route_*, the newest replay per route record."""
    jobs = []
    for route_dir in sorted(Path(run_dir, "routes").glob("route_*")):
        result_path = route_dir / "result.json"
        if not result_path.is_file():
            print(f"[{route_dir.name}] no result.json, skipped")
            continue
        with open(result_path) as result_file:
            records = json.load(result_file)["_checkpoint"]["records"]
        replays = newest_files(route_dir, "obs_html", ".replay.zlib", len(records))
        if len(replays) != len(records):
            print(f"[{route_dir.name}] {len(records)} route records but {len(replays)} replays; pairing the newest")
        if not replays:
            continue
        records = records[-len(replays) :]
        world_logs = newest_files(route_dir, "world_log", ".npz", len(replays))
        for k, (record, replay) in enumerate(zip(records, replays)):
            town = towns.get(route_index_rep(record)[0])
            if town is None and k < len(world_logs):
                town = world_log_town(world_logs[k])
            jobs.append((record, replay, town or ""))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="output dir of 11_carla_longest6.sh (holds routes/route_*/)")
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
    pages = [(obs_dir / page_name(record, town), record, replay, town) for record, replay, town in jobs]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(lambda job: render_page(job[2], job[0], score_panel(job[1], job[3])), pages))
    file_metrics = {page.name: gallery_metrics(record) for page, record, _, _ in pages}
    report = run_dir / "report" / "index.html"
    links = [("Analysis report", os.path.relpath(report, obs_dir))] if report.is_file() else []
    viz.build_gallery_index(str(obs_dir), file_metrics=file_metrics, links=links)
    print(f"[render_carla_obs_html] {len(pages)} route pages rendered -> {obs_dir / 'index.html'}")


if __name__ == "__main__":
    main()
