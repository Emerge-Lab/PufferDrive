"""Attribute each logged agent's first infraction in training-render replays (.replay.zlib) to its circumstances.

usage: python scripts/attribute_training_infractions.py <run>/renders/<epoch>/replays [dt]
(needs eval.keep_zlib_replays=true on the render)
"""
import glob
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pufferlib.viz import read_replay_zlib  # noqa: E402

COLLISION_IDX, OFFROAD_IDX, RED_LIGHT_IDX, STOP_SIGN_IDX, LANE_DIST_IDX = 0, 1, 2, 3, 5
I_TYPE, I_ACTIVE, I_STOPPED, I_REMOVED, I_SLOT, I_BLIND, I_PHANTOM = 1, 3, 4, 5, 7, 8, 9
F_X, F_Y, F_HEADING, F_LENGTH, F_WIDTH, F_SPEED = 0, 1, 3, 4, 5, 6
PERTURBATION_LOOKBACK_FRAMES = 10  # 3 s bouts at dt 0.3
SPAWN_FRAMES = 10


def first_frame(flags):
    hits = np.flatnonzero(flags)
    return int(hits[0]) if hits.size else None


def classify_collision(f32, i32, m, t, a):
    pos = f32[t][:, [F_X, F_Y]]
    valid = i32[t, :, 2] == 1
    valid[a] = False
    dist = np.linalg.norm(pos - pos[a], axis=1)
    dist[~valid] = np.inf
    # Prefer the agent whose collision flag fires in the same frame; fall back to the nearest body.
    co_flagged = valid & (m[t, :, COLLISION_IDX] == 1)
    if co_flagged.any():
        dist_co = np.where(co_flagged, dist, np.inf)
        p = int(np.argmin(dist_co))
    else:
        p = int(np.argmin(dist))
    gap = dist[p] - 0.5 * (f32[t, a, F_LENGTH] + f32[t, p, F_LENGTH])
    if not np.isfinite(dist[p]) or gap > 2.0:
        return "no partner found", None
    t0 = max(0, t - PERTURBATION_LOOKBACK_FRAMES)
    tp = t - 1 if t > 0 else t
    tags = []
    if i32[t0 : t + 1, p, I_BLIND].any():
        tags.append("partner blind now")
    if i32[t0 : t + 1, p, I_PHANTOM].any():
        tags.append("partner phantom-braking now")
    if not tags and (i32[:, p, I_BLIND].any() or i32[:, p, I_PHANTOM].any()):
        tags.append("partner is a designated perturber (bout elsewhere)")
    if t > 0 and i32[t - 1, p, I_STOPPED] == 1:
        tags.append("partner parked by earlier infraction")
    elif f32[tp, p, F_SPEED] < 0.3:
        tags.append("partner stationary")
    if i32[t, p, I_ACTIVE] == 0:
        tags.append("partner not policy-controlled")
    if not tags:
        if m[t, p, COLLISION_IDX] == 1 and first_frame(m[:, p, COLLISION_IDX]) == t:
            tags.append("both moving, mutual first collision")
        else:
            tags.append("both moving, other")
    dh = abs((f32[t, a, F_HEADING] - f32[t, p, F_HEADING] + np.pi) % (2 * np.pi) - np.pi)
    rel = pos[p] - pos[a]
    ahead = np.cos(f32[t, a, F_HEADING]) * rel[0] + np.sin(f32[t, a, F_HEADING]) * rel[1] > 0
    if dh < np.radians(30):
        geometry = "rear-end, ego behind" if ahead else "rear-end, ego in front"
    elif dh > np.radians(150):
        geometry = "head-on"
    else:
        geometry = "crossing/side"
    detail = f"ego {f32[tp, a, F_SPEED]:.1f} m/s, partner {p} {f32[tp, p, F_SPEED]:.1f} m/s, {geometry}"
    return " + ".join(tags), detail


def analyze(path, dt):
    _, c = read_replay_zlib(path, chunk_names=["agent_f32", "agent_i32", "metrics_f32"])
    f32, i32, m = c["agent_f32"], c["agent_i32"], c["metrics_f32"]
    frames, agents = m.shape[:2]
    rows = []
    logged = 0
    for a in range(agents):
        if i32[0, a, I_ACTIVE] != 1:
            continue
        perturbed = bool(i32[:, a, I_BLIND].any() or i32[:, a, I_PHANTOM].any())
        if not perturbed:
            logged += 1
        events = {
            "collision": first_frame(m[:, a, COLLISION_IDX]),
            "offroad": first_frame(m[:, a, OFFROAD_IDX]),
            "red_light": first_frame(m[:, a, RED_LIGHT_IDX]),
            "stop_sign": first_frame(m[:, a, STOP_SIGN_IDX]),
        }
        events = {k: v for k, v in events.items() if v is not None}
        if not events:
            continue
        kind = min(events, key=events.get)
        t = events[kind]
        tp = max(t - 1, 0)
        speed = f32[tp, a, F_SPEED]
        if kind == "collision":
            tag, detail = classify_collision(f32, i32, m, t, a)
        elif kind == "offroad":
            tag = "offroad within spawn frames" if t < SPAWN_FRAMES else ("offroad at creep speed (<1 m/s)" if speed < 1.0 else ("offroad slow (1-4 m/s)" if speed < 4.0 else "offroad fast (>4 m/s)"))
            detail = f"{speed:.1f} m/s, lane dist {m[tp, a, LANE_DIST_IDX]:.1f} m"
        else:
            tag = kind
            detail = f"{speed:.1f} m/s"
        rows.append((kind, tag, detail, t * dt, a, perturbed))
    return frames, agents, logged, rows


def main():
    replays_dir = sys.argv[1]
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    paths = sorted(glob.glob(os.path.join(replays_dir, "*.replay.zlib")))
    totals = Counter()
    logged_total = 0
    for path in paths:
        frames, agents, logged, rows = analyze(path, dt)
        logged_total += logged
        print(f"\n== {os.path.basename(path)}: {frames} frames, {agents} agents ({logged} logged policy agents)")
        for kind, tag, detail, t_s, a, perturbed in sorted(rows, key=lambda r: r[3]):
            flag = " [ego perturbed, not logged]" if perturbed else ""
            print(f"  t={t_s:6.1f}s agent {a:3d} {kind:9s} {tag}: {detail}{flag}")
            if not perturbed:
                totals[(kind, tag)] += 1
    n_inf = sum(totals.values())
    print(f"\n== totals over {len(paths)} episodes, {logged_total} logged agent-episodes, {n_inf} first infractions"
          f" ({n_inf / max(logged_total, 1):.3f} per agent-episode)")
    for (kind, tag), n in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {kind:9s} {tag}")


if __name__ == "__main__":
    main()
