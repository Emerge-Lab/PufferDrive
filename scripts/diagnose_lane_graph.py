#!/usr/bin/env python3
"""Diagnose why generate_random_route dead-ends on nuplan lane graphs.

Loads one .bin via Drive(), reads the lane graph from get_state(), mimics
generate_random_route's walk (random successor, avoid-visited), and reports:
  - lane count + num_exits distribution
  - dead-end ratio (lanes with num_exits == 0)
  - reachable-distance distribution starting from every lane
  - fraction of lanes that can hit 360 m (the static target_type threshold)
"""

import argparse
import math
import os
import random
import sys

from pufferlib.ocean.drive.drive import Drive


def is_road_lane(t):
    return 0 <= t <= 9


def lane_length(road):
    """Sum of segment lengths between consecutive geometry points."""
    xs, ys = road.get("x"), road.get("y")
    if not xs or not ys or len(xs) < 2:
        return 0.0
    total = 0.0
    for i in range(len(xs) - 1):
        total += math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
    return total


def random_walk_distance(roads, start_idx, target_distance=360.0, max_route_length=64, rng=None):
    """Mimic generate_random_route. Returns (achieved_distance, route_length, dead_end_reason)."""
    rng = rng or random
    visited = {start_idx}
    route_len = 1
    accumulated = lane_length(roads[start_idx])
    current = start_idx
    while accumulated < target_distance and route_len < max_route_length:
        r = roads[current]
        exits = [e for e in (r.get("exit_lanes") or []) if e != -1 and e not in visited]
        if not exits:
            return accumulated, route_len, "dead_end"
        nxt = rng.choice(exits)
        visited.add(nxt)
        route_len += 1
        accumulated += lane_length(roads[nxt])
        current = nxt
    if accumulated >= target_distance:
        return accumulated, route_len, "reached"
    return accumulated, route_len, "max_route_length"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True, help="Path to one .bin (or its parent dir)")
    p.add_argument("--target-distance", type=float, default=360.0)
    p.add_argument("--samples-per-lane", type=int, default=5, help="random walks per starting lane")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    map_dir = args.bin if os.path.isdir(args.bin) else os.path.dirname(args.bin)
    map_filename = None if os.path.isdir(args.bin) else os.path.basename(args.bin)

    env = Drive(
        map_dir=map_dir,
        num_maps=1 if map_filename else 8,
        num_agents=1,
        max_agents_per_env=1,
        min_agents_per_env=1,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        init_steps=10,
        scenario_length=91,
        seed=args.seed,
    )
    env.reset(seed=args.seed)
    state = env.get_state()

    # state["env_state"] is a list per Drive env; pick first
    env_states = state if isinstance(state, list) else state.get("env_state", state)
    if isinstance(env_states, list):
        first = env_states[0]
    else:
        first = env_states
    roads = first.get("road_elements") or first.get("roads") or []
    print(f"Loaded {len(roads)} road elements", flush=True)

    lanes = [(i, r) for i, r in enumerate(roads) if is_road_lane(r.get("type", -1))]
    print(f"Drivable lane count: {len(lanes)}")

    exits_hist = {}
    for i, r in lanes:
        n = len(r.get("exit_lanes") or [])
        exits_hist[n] = exits_hist.get(n, 0) + 1
    print("num_exits distribution:")
    for k in sorted(exits_hist):
        print(f"  num_exits={k}: {exits_hist[k]} lanes")

    dead_end_lanes = [i for i, r in lanes if not (r.get("exit_lanes") or [])]
    print(f"\nDead-end lanes (num_exits==0): {len(dead_end_lanes)} / {len(lanes)} = {100.0 * len(dead_end_lanes) / max(1, len(lanes)):.1f}%")

    lengths = [lane_length(r) for _, r in lanes]
    if lengths:
        lengths.sort()
        n = len(lengths)
        print(f"\nLane length stats (m):")
        print(f"  min   = {lengths[0]:.1f}")
        print(f"  p10   = {lengths[n // 10]:.1f}")
        print(f"  p50   = {lengths[n // 2]:.1f}")
        print(f"  p90   = {lengths[(9 * n) // 10]:.1f}")
        print(f"  max   = {lengths[-1]:.1f}")
        print(f"  mean  = {sum(lengths) / n:.1f}")
        print(f"  total = {sum(lengths):.1f}")

    print(f"\nRandom-walk results, target={args.target_distance}m, samples={args.samples_per_lane}/lane")
    outcomes = {"reached": 0, "dead_end": 0, "max_route_length": 0}
    achieved = []
    rlens = []
    for i, _ in lanes:
        for _ in range(args.samples_per_lane):
            dist, rlen, why = random_walk_distance(
                roads, i, target_distance=args.target_distance, rng=random
            )
            outcomes[why] += 1
            achieved.append(dist)
            rlens.append(rlen)

    total = sum(outcomes.values())
    for k, v in outcomes.items():
        print(f"  {k:20s}: {v} ({100.0 * v / max(1, total):.1f}%)")

    achieved.sort()
    rlens.sort()
    if achieved:
        n = len(achieved)
        print(f"  achieved distance percentiles (m): "
              f"p10={achieved[n // 10]:.0f}, p50={achieved[n // 2]:.0f}, "
              f"p90={achieved[(9 * n) // 10]:.0f}, max={achieved[-1]:.0f}")
        print(f"  route length percentiles (lanes): "
              f"p10={rlens[n // 10]}, p50={rlens[n // 2]}, "
              f"p90={rlens[(9 * n) // 10]}, max={rlens[-1]}")


if __name__ == "__main__":
    main()
