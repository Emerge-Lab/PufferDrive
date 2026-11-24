"""Replay a PufferDrive log so a dataset can be checked by eye.

This is the shared inspector for every data/<name> directory. It
replays two agents and writes both files under data/<name>/assets.
The PNG is a 5-second contact sheet: a cropped bird's-eye view beside
the observation that agent would receive. The GIF is the same pair of
views for 10 seconds, or for the whole log when that log is shorter.
The observation is read from the
simulator, then drawn by undoing the scales in ocean/drive/constants.h.
Matplotlib draws the boxes. Seaborn sets the theme and the agent colors.

  python utils/sanity.py --data-dir data/i24
  python utils/sanity.py --data-dir data/womd_train_1000
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.common import ROAD_LANE

# Matches ocean/drive/constants.h and compute_observations.
EGO_FEATURES = 9
PARTNER_FEATURES = 7
ROAD_FEATURES = 7
MAX_AGENTS = 64
MAX_ROADS = 75
OBS_SIZE = (
    EGO_FEATURES
    + PARTNER_FEATURES * MAX_AGENTS
    + ROAD_FEATURES * MAX_ROADS
)
MAX_VEH_LEN = 30.0
MAX_VEH_WIDTH = 15.0
MAX_SPEED = 80.0
MAX_ROAD_SEGMENT_LENGTH = 100.0
OBS_GOAL_SCALE = 0.005
OBS_SPEED_SCALE = 1.0 / MAX_SPEED
OBS_POSITION_SCALE = 0.02
OBS_ACCEL_SCALE = 0.05
OBS_YAW_SCALE = 0.5
SAMPLE_HZ = 10
REPLAY_SECONDS = 5
VIDEO_SECONDS = 10
FIGURE_DPI = 200
CROP_M = 40.0
# load_map_binary and the offline library both stop at this many bodies.
DRIVE_AGENT_MAX = 256

sns.set_theme(style="ticks", context="notebook")
_AGENTS = sns.color_palette("deep", 2)
EGO_COLOR = _AGENTS[0]
PARTNER_COLOR = _AGENTS[1]
ROAD_COLOR = {
    0: "#b0b6bd",
    1: "#e6c35c",
    2: "#222222",
    3: "#d62728",
    4: "#4c78a8",
    5: "#9467bd",
    6: "#8c564b",
}


def load_lib():
    lib = ctypes.CDLL(str(_REPO / "build" / "libdrive_offline.so"))
    handle = ctypes.c_void_p
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)

    def fn(name, restype, argtypes):
        f = getattr(lib, name)
        f.restype = restype
        f.argtypes = argtypes
        return f

    fn("pf_create_scenario", handle, [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
    ])
    fn("pf_close", None, [handle])
    fn("pf_place", None, [handle, ctypes.c_int])
    fn("pf_recompute", None, [handle])
    fn("pf_obs", c_float_p, [handle])
    fn("pf_obs_size", ctypes.c_int, [handle])
    fn("pf_n", ctypes.c_int, [handle])
    fn("pf_log_n", ctypes.c_int, [handle])
    fn("pf_log_traj", ctypes.c_int, [handle])
    fn("pf_log_valid", ctypes.c_int, [handle, ctypes.c_int, ctypes.c_int])
    fn("pf_agent", None, [handle, ctypes.c_int, c_float_p])
    fn("pf_road_counts", None, [handle, c_int_p, c_int_p])
    fn("pf_roads", None, [handle, c_int_p, c_int_p, c_float_p, c_float_p])
    return lib


def open_scene(lib, data_dir: Path, map_id: int, json_path: Path):
    handle = lib.pf_create_scenario(
        str(data_dir).encode(), int(map_id), str(json_path).encode(),
    )
    if not handle:
        raise RuntimeError(f"pf_create_scenario failed for {json_path.name}")
    if int(lib.pf_obs_size(handle)) != OBS_SIZE:
        lib.pf_close(handle)
        raise RuntimeError("observation width does not match constants.h")
    return handle


def scene_frame(lib, handle):
    n = int(lib.pf_n(handle))
    agents = np.zeros((n, 11), dtype=np.float32)
    for i in range(n):
        lib.pf_agent(handle, i, agents[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    nroads = ctypes.c_int()
    npts = ctypes.c_int()
    lib.pf_road_counts(handle, ctypes.byref(nroads), ctypes.byref(npts))
    offsets = np.zeros(int(nroads.value) + 1, dtype=np.int32)
    types = np.zeros(int(nroads.value), dtype=np.int32)
    xs = np.zeros(int(npts.value), dtype=np.float32)
    ys = np.zeros(int(npts.value), dtype=np.float32)
    lib.pf_roads(
        handle,
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        types.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ys.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    obs_n = int(lib.pf_obs_size(handle))
    obs = np.ctypeslib.as_array(lib.pf_obs(handle), shape=(n, obs_n)).copy()
    return agents, offsets, types, xs, ys, obs


def _box(ax, x, y, heading, length, width, **kwargs):
    c = math.cos(heading)
    s = math.sin(heading)
    hl = length * 0.5
    hw = width * 0.5
    corners = []
    for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw), (hl, hw)):
        corners.append((x + dx * c - dy * s, y + dx * s + dy * c))
    ax.add_patch(Polygon(corners, closed=True, **kwargs))


def draw_bev(ax, agents, offsets, types, xs, ys, ego):
    ex, ey, eh = float(agents[ego, 0]), float(agents[ego, 1]), float(agents[ego, 2])
    for r in range(len(types)):
        a = int(offsets[r])
        b = int(offsets[r + 1])
        kind = int(types[r]) - ROAD_LANE
        ax.plot(xs[a:b], ys[a:b], color=ROAD_COLOR.get(kind, "#777777"),
            linewidth=1.0, solid_capstyle="round")
    for i, agent in enumerate(agents):
        if agent[10] < 0.5:
            continue
        color = EGO_COLOR if i == ego else PARTNER_COLOR
        _box(ax, agent[0], agent[1], agent[2], agent[8], agent[7],
            facecolor=color, edgecolor="black", linewidth=0.4, alpha=0.9, zorder=3)
    ax.annotate("", xy=(ex + math.cos(eh) * 4, ey + math.sin(eh) * 4),
        xytext=(ex, ey), arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        zorder=4)
    ax.set_xlim(ex - CROP_M, ex + CROP_M)
    ax.set_ylim(ey - CROP_M, ey + CROP_M)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_observation(ax, obs):
    """Ego frame. Forward is up. Left is left.

    Roads are the points in the observation, one marker per slot, not a
    rebuilt segment. The ego box is its length and width. The star is
    the relative goal, the arrow is its speed, and the corner text is
    the rest of the ego vector from Appendix A of arXiv 2502.14706,
    plus the accel, yaw rate, and target speed this simulator adds.
    """
    goal_x = float(obs[0]) / OBS_GOAL_SCALE
    goal_y = float(obs[1]) / OBS_GOAL_SCALE
    speed = float(obs[2]) / OBS_SPEED_SCALE
    width = float(obs[3]) * MAX_VEH_WIDTH
    length = float(obs[4]) * MAX_VEH_LEN
    collision = float(obs[5]) > 0.5
    accel = float(obs[6]) / OBS_ACCEL_SCALE
    yaw = float(obs[7]) / OBS_YAW_SCALE
    target = float(obs[8]) / OBS_SPEED_SCALE

    road0 = EGO_FEATURES + PARTNER_FEATURES * MAX_AGENTS
    for i in range(MAX_ROADS):
        slot = obs[road0 + i * ROAD_FEATURES: road0 + (i + 1) * ROAD_FEATURES]
        if float(np.max(np.abs(slot[:2]))) == 0.0:
            continue
        mx = float(slot[0]) / OBS_POSITION_SCALE
        my = float(slot[1]) / OBS_POSITION_SCALE
        kind = int(round(float(slot[6])))
        ax.scatter(-my, mx, s=14, c=ROAD_COLOR.get(kind, "#777777"),
            marker="o", linewidths=0, zorder=2)
    base = EGO_FEATURES
    for i in range(MAX_AGENTS):
        slot = obs[base + i * PARTNER_FEATURES: base + (i + 1) * PARTNER_FEATURES]
        if float(np.max(np.abs(slot))) == 0.0:
            continue
        px = float(slot[0]) / OBS_POSITION_SCALE
        py = float(slot[1]) / OBS_POSITION_SCALE
        heading = math.atan2(float(slot[5]), float(slot[4])) + math.pi / 2
        _box(ax, -py, px, heading, float(slot[3]) * MAX_VEH_LEN, float(slot[2]) * MAX_VEH_WIDTH,
            facecolor=PARTNER_COLOR, edgecolor="black", linewidth=0.4, alpha=0.9, zorder=3)
    edge = "#d62728" if collision else "black"
    _box(ax, 0.0, 0.0, math.pi / 2, length, width,
        facecolor=EGO_COLOR, edgecolor=edge, linewidth=1.0 if collision else 0.4, zorder=4)
    ax.plot(-goal_y, goal_x, marker="*", color="#d62728", markersize=8,
        linestyle="none", zorder=5)
    if speed > 0.2:
        ax.annotate("", xy=(0.0, min(speed * 0.35, 12.0)), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=EGO_COLOR, lw=1.2), zorder=5)
    goal_m = math.hypot(goal_x, goal_y)
    caption = f"v {speed:.1f} m/s   goal {goal_m:.0f} m   {length:.1f}×{width:.1f} m"
    if collision:
        caption += "   collision"
    caption += f"\na {accel:+.1f}   yaw {yaw:+.2f}   target {target:.1f}"
    ax.text(0.02, 0.02, caption, transform=ax.transAxes, fontsize=6, va="bottom",
        ha="left", color="#222222", linespacing=1.15,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88),
        zorder=6)
    ax.set_xlim(-CROP_M, CROP_M)
    ax.set_ylim(-CROP_M * 0.35, CROP_M)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _eligible(lib, handle, frames):
    n = int(lib.pf_log_n(handle))
    chosen = []
    for i in range(n):
        if all(lib.pf_log_valid(handle, i, t) for t in frames):
            chosen.append(i)
    return chosen


def still_frames():
    return list(range(0, REPLAY_SECONDS * SAMPLE_HZ + 1, SAMPLE_HZ))


def clip_frames(state_count: int) -> list[int]:
    """10 Hz frames from t = 0 through the shorter of 10 s and the log."""
    count = min(state_count, VIDEO_SECONDS * SAMPLE_HZ)
    return list(range(count))


def pick_scene(data_dir: Path, seed: int):
    jsons = sorted((data_dir / "json").glob("*.json"))
    if not jsons:
        raise SystemExit(f"no json in {data_dir / 'json'}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(jsons))
    sheet = still_frames()
    for index in order:
        path = jsons[int(index)]
        with path.open() as handle:
            scenario = json.load(handle)
        objects = scenario.get("objects") or []
        if not 2 <= len(objects) <= DRIVE_AGENT_MAX:
            continue
        states = len(objects[0].get("position") or [])
        if states <= sheet[-1]:
            continue
        frames = clip_frames(states)
        vehicles = []
        for i, obj in enumerate(objects):
            if obj.get("type") not in ("vehicle", "truck"):
                continue
            valid = obj.get("valid") or []
            if len(valid) < states:
                continue
            if all(bool(valid[t]) for t in frames):
                vehicles.append(i)
        if len(vehicles) < 2:
            continue
        agents = [int(v) for v in rng.choice(vehicles, size=2, replace=False)]
        return int(index), path, agents, frames
    raise SystemExit(
        f"no scene in {data_dir} has two vehicles over {REPLAY_SECONDS} s"
    )


def _sheet(columns, agents, frames, data_dir, map_id, json_path, seed):
    fig, axes = plt.subplots(
        len(agents) * 2, len(frames),
        figsize=(2.4 * len(frames), 2.3 * len(agents) * 2),
        squeeze=False,
    )
    for row, agent in enumerate(agents):
        for col, frame in enumerate(frames):
            agents_xy, offsets, types, xs, ys, obs = columns[frame]
            bev = axes[row * 2, col]
            view = axes[row * 2 + 1, col]
            draw_bev(bev, agents_xy, offsets, types, xs, ys, agent)
            draw_observation(view, obs[agent])
            if row == 0:
                bev.set_title(f"{frame / SAMPLE_HZ:.0f} s", fontsize=9)
            if col == 0:
                bev.set_ylabel(f"agent {agent}\nbird's eye", fontsize=8)
                view.set_ylabel("observation", fontsize=8)
    fig.suptitle(
        f"{data_dir.name} map {map_id}  {json_path.name}  seed {seed}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _write_clip(path, columns, agents, frames, data_dir, map_id, json_path):
    fig, axes = plt.subplots(
        2, len(agents), figsize=(4.0 * len(agents), 7.0), squeeze=False,
    )
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.04, top=0.88, wspace=0.08, hspace=0.18)
    writer = PillowWriter(fps=SAMPLE_HZ)
    with writer.saving(fig, str(path), dpi=FIGURE_DPI):
        for frame in frames:
            agents_xy, offsets, types, xs, ys, obs = columns[frame]
            for row, agent in enumerate(agents):
                bev = axes[0, row]
                view = axes[1, row]
                bev.clear()
                view.clear()
                draw_bev(bev, agents_xy, offsets, types, xs, ys, agent)
                draw_observation(view, obs[agent])
                bev.set_title(f"agent {agent}  bird's eye", fontsize=9)
                view.set_title("observation", fontsize=9)
            fig.suptitle(
                f"{data_dir.name} map {map_id}  {json_path.name}  "
                f"{frame / SAMPLE_HZ:.1f} s",
                fontsize=11,
            )
            writer.grab_frame()
    plt.close(fig)


def render(
    data_dir: Path,
    out_dir: Path,
    seed: int,
    map_id: int | None = None,
    agents: list[int] | None = None,
    steps: int | None = None,
):
    jsons = sorted((data_dir / "json").glob("*.json"))
    if map_id is None:
        map_id, json_path, picked, frames = pick_scene(data_dir, seed)
        if agents is None:
            agents = picked
    else:
        if not 0 <= map_id < len(jsons):
            raise SystemExit(f"map {map_id} is outside {data_dir / 'json'}")
        json_path = jsons[map_id]
        with json_path.open() as handle:
            scenario = json.load(handle)
        states = len((scenario.get("objects") or [{}])[0].get("position") or [])
        count = states if steps is None else min(states, steps)
        if count < 1:
            raise SystemExit(f"{json_path.name} has no states")
        frames = list(range(count))
        if agents is None:
            _, _, agents, _ = pick_scene(data_dir, seed)
    if steps is not None:
        frames = frames[:steps]
    lib = load_lib()
    handle = open_scene(lib, data_dir, map_id, json_path)
    try:
        columns = {}
        with json_path.open() as file:
            state_count = len(json.load(file)["objects"][0]["position"])
        # The sheet stays one frame per second. A short GIF does not drop it.
        sheet = [t for t in still_frames() if t < state_count]
        for frame in sorted(set(frames) | set(sheet)):
            lib.pf_place(handle, frame)
            lib.pf_recompute(handle)
            columns[frame] = scene_frame(lib, handle)
    finally:
        lib.pf_close(handle)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{data_dir.name}_map_{map_id:03d}_sanity"
    sheet_fig = _sheet(
        columns, agents, sheet, data_dir, map_id, json_path, seed,
    )
    png = out_dir / f"{stem}.png"
    sheet_fig.savefig(png, dpi=FIGURE_DPI)
    plt.close(sheet_fig)
    gif = out_dir / f"{stem}.gif"
    _write_clip(gif, columns, agents, frames, data_dir, map_id, json_path)
    return png, gif, map_id, json_path, agents, len(frames) / SAMPLE_HZ


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=_REPO / "data" / "womd_train_1000")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Default: <data-dir>/assets",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-id", type=int, default=None)
    parser.add_argument(
        "--agents", type=str, default=None,
        help="Comma-separated agent indices. Default: two vehicles from --seed.",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="GIF length in 10 Hz steps. Default: 10 seconds, or the whole log.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or (args.data_dir / "assets")
    agents = None
    if args.agents is not None:
        agents = [int(part) for part in args.agents.split(",") if part.strip()]
    png, gif, map_id, json_path, agents, video_seconds = render(
        args.data_dir, out_dir, args.seed,
        map_id=args.map_id, agents=agents, steps=args.steps,
    )
    print(json.dumps({
        "png": str(png),
        "gif": str(gif),
        "map_id": map_id,
        "json": json_path.name,
        "agents": agents,
        "sheet_seconds": REPLAY_SECONDS,
        "video_seconds": video_seconds,
    }, indent=2))


if __name__ == "__main__":
    main()
