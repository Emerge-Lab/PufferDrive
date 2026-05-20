#!/usr/bin/env python3
"""Quick probe: trajectory_length distribution across nuplan bins."""
import argparse
import os
import struct


def probe(path):
    with open(path, "rb") as f:
        data = f.read()
    # The bin format starts with a number of road elements, then road element
    # data, then agents. We can't easily seek without parsing — but each
    # agent record has trajectory_length as one of its leading ints. Easiest
    # to just load via Drive() and read agent.trajectory_length.
    from pufferlib.ocean.drive.drive import Drive
    map_dir = os.path.dirname(path)
    env = Drive(
        map_dir=map_dir,
        num_maps=1,
        num_agents=1,
        max_agents_per_env=1,
        min_agents_per_env=1,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        init_steps=10,
        scenario_length=200,  # ask for more steps so we don't artificially cap
        seed=42,
    )
    env.reset(seed=42)
    state = env.get_state()
    # state is a list of per-env states
    first = state[0] if isinstance(state, list) else state
    agents = first.get("agents") or first.get("active_agents") or []
    lengths = [a.get("trajectory_length", 0) for a in agents if isinstance(a, dict)]
    return lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--samples", type=int, default=10)
    args = p.parse_args()
    bins = sorted(b for b in os.listdir(args.dir) if b.endswith(".bin"))
    print(f"Total bins: {len(bins)}")
    sampled = bins[: args.samples]
    all_lengths = []
    for b in sampled:
        path = os.path.join(args.dir, b)
        try:
            lens = probe(path)
            print(f"  {b}: agent trajectory_lengths = {sorted(set(lens))[:10]}")
            all_lengths.extend(lens)
        except Exception as e:
            print(f"  {b}: error {e}")
    if all_lengths:
        all_lengths.sort()
        n = len(all_lengths)
        print(
            f"\nAggregate ({len(sampled)} bins, {n} agents): min={all_lengths[0]}, "
            f"p50={all_lengths[n // 2]}, p90={all_lengths[(9 * n) // 10]}, max={all_lengths[-1]}"
        )


if __name__ == "__main__":
    main()
