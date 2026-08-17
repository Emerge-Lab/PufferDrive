#!/usr/bin/env python3
"""scan_env_obs.py — Probe eval-env observations map-by-map for non-finite or
extreme values.

Rebuilds the exact eval env from a run's resolved_benchmark.yaml, then for each
map index resets a one-scenario env and steps it a few times, reporting any
non-finite observation/reward and the max |obs| seen. Finds maps whose runtime
obs math misbehaves even when the map file itself scans clean.

Usage:
    python data_utils/scan_env_obs.py <resolved_benchmark.yaml> <start_map_idx> <end_map_idx>
"""

import sys

import numpy as np
import yaml

from pufferlib.ocean.drive.drive import Drive

PROBE_STEPS = 5
EXTREME_OBS_THRESHOLD = 50.0
VALID_COUNT_FEATURES = 4  # trailing slot-count features are raw counts; exempt from the extreme check


def probe_map(env_config, map_idx):
    kwargs = dict(env_config)
    kwargs["eval_mode"] = 1
    kwargs["starting_map"] = map_idx
    kwargs["num_eval_scenarios"] = 1
    kwargs["resample_frequency"] = kwargs.get("scenario_length") or 1000
    kwargs["capture_replay"] = False
    env = Drive(**kwargs)
    try:
        map_name = env.map_files[map_idx].split("/")[-1]
        observations, _ = env.reset()
        worst_abs = 0.0
        nonfinite_count = 0
        for step_idx in range(PROBE_STEPS + 1):
            finite_mask = np.isfinite(observations)
            nonfinite_count += int((~finite_mask).sum())
            normalized_features = observations[:, : -VALID_COUNT_FEATURES]
            normalized_finite = np.isfinite(normalized_features)
            if normalized_finite.any():
                worst_abs = max(worst_abs, float(np.abs(normalized_features[normalized_finite]).max()))
            if step_idx == PROBE_STEPS:
                break
            actions = np.zeros((observations.shape[0], *env.single_action_space.shape), dtype=np.float32)
            if step_idx % 2 == 1:
                actions[:] = 1.0
            observations, rewards, _, _, _ = env.step(actions)
            nonfinite_count += int((~np.isfinite(rewards)).sum())
        return map_name, nonfinite_count, worst_abs
    finally:
        env.close()


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    resolved_path, start_idx, end_idx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    env_config = yaml.safe_load(open(resolved_path))["args"]["env"]

    bad_map_count = 0
    for map_idx in range(start_idx, end_idx):
        try:
            map_name, nonfinite_count, worst_abs = probe_map(env_config, map_idx)
        except Exception as probe_error:
            bad_map_count += 1
            print(f"[{map_idx}] PROBE ERROR: {probe_error}", flush=True)
            continue
        if nonfinite_count:
            bad_map_count += 1
            print(f"[{map_idx}] {map_name}: {nonfinite_count} NON-FINITE values, max|obs|={worst_abs:.1f}", flush=True)
        elif worst_abs > EXTREME_OBS_THRESHOLD:
            bad_map_count += 1
            print(f"[{map_idx}] {map_name}: EXTREME obs, max|obs|={worst_abs:.1f}", flush=True)
        elif map_idx % 50 == 0:
            print(f"[{map_idx}] {map_name}: ok, max|obs|={worst_abs:.1f}", flush=True)
    print(f"maps {start_idx}..{end_idx}: {bad_map_count} with problems", flush=True)
    sys.exit(1 if bad_map_count else 0)


if __name__ == "__main__":
    main()
