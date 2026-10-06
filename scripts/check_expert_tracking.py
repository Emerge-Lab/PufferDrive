"""Measure how well the expert_tracking controller reproduces human logs on the sim dynamics.

Runs many short sub-episodes from random start steps and reports the per-step
position error against the log (so a reset transient shows up at steps 1-3),
the ADE/FDE distribution over sub-episodes, and the label histogram. Compare
against --sdc-controller policy (constant zero jerk) to see the fit's headroom.

    python scripts/check_expert_tracking.py --map-dir <replay .bin dir> --num-maps 100
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.drive import binding  # noqa: E402
from pufferlib.ocean.drive.drive import Drive  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=32, help="sub-episodes per round")
    parser.add_argument("--rounds", type=int, default=4, help="rounds of fresh random start steps")
    parser.add_argument("--horizon", type=int, default=22, help="steps per sub-episode")
    parser.add_argument("--scenario-length", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--sdc-controller", default="expert_tracking", choices=["expert_tracking", "policy"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--error-threshold-m", type=float, default=0.5)
    return parser.parse_args()


def make_env(args, seed):
    return Drive(
        num_agents=args.num_envs,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=args.num_maps,
        map_dir=args.map_dir,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller=args.sdc_controller,
        non_sdc_controller="replay",
        non_vehicle_controller="replay",
        goal_source="gt",
        action_type="discrete",
        dynamics_model="jerk",
        dt=args.dt,
        scenario_length=args.scenario_length,
        init_step_spread=True,
        init_step_min_horizon=args.horizon + 1,
        resample_frequency=1_000_000,
        termination_mode=False,
        collision_behavior="ignore",
        offroad_behavior="ignore",
        traffic_light_behavior="ignore",
        stop_sign_behavior="ignore",
        compute_eval_metrics=True,
        capture_replay=True,
        report_interval=1,
        seed=seed,
    )


def run_round(env, horizon):
    """Returns per-env running-ADE curves [num_envs, horizon + 1], labels [horizon, num_envs], masks [horizon, num_envs]."""
    env.reset()
    placeholder = np.zeros_like(env.actions)
    labels = np.empty((horizon, env.num_agents), dtype=np.int64)
    masks = np.empty((horizon, env.num_agents), dtype=bool)
    for step in range(horizon):
        env.step(placeholder)
        labels[step] = env.actions.reshape(-1)
        masks[step] = env.masks.reshape(-1) != 0
    env._capture_replay_step()

    running_ade = np.empty((env.num_agents, horizon + 1), dtype=np.float64)
    for env_idx, capture in enumerate(env._replay_captures):
        frames = capture["frames"]["metrics_f32"]
        assert len(frames) == horizon + 1, f"expected {horizon + 1} frames, got {len(frames)}"
        ego_rows = capture["frames"]["agent_i32"][0][:, 7]
        ego_row = int(np.flatnonzero(ego_rows == 0)[0])
        running_ade[env_idx] = [frame[ego_row, binding.AVG_DISPLACEMENT_ERROR_IDX] for frame in frames]
    return running_ade, labels, masks


def per_step_errors(running_ade):
    """Recovers the per-step displacement from the running mean; step k uses frames k-1 and k."""
    steps = np.arange(running_ade.shape[1])
    cumulative = running_ade * steps
    return np.diff(cumulative, axis=1)


def main():
    args = parse_args()
    curves = []
    label_counter = Counter()
    invalid_labels = 0
    total_labels = 0
    for round_idx in range(args.rounds):
        env = make_env(args, args.seed + round_idx)
        try:
            running_ade, labels, masks = run_round(env, args.horizon)
        finally:
            env.close()
        curves.append(per_step_errors(running_ade))
        label_counter.update(labels[masks].tolist())
        invalid_labels += int((~masks).sum())
        total_labels += masks.size

    errors = np.concatenate(curves, axis=0)
    ade = errors.mean(axis=1)
    fde = errors[:, -1]
    peak = errors.max(axis=1)

    print(f"sub-episodes: {errors.shape[0]}  horizon: {args.horizon} steps at dt={args.dt}  sdc={args.sdc_controller}")
    print("per-step position error (m):")
    print("  step   mean    p50    p95    max")
    for step in range(args.horizon):
        column = errors[:, step]
        print(
            f"  {step + 1:4d} {column.mean():6.3f} {np.median(column):6.3f} {np.percentile(column, 95):6.3f} {column.max():6.3f}"
        )
    print("summary (m):")
    for name, values in (("ADE", ade), ("FDE", fde), ("peak", peak)):
        print(
            f"  {name}: mean={values.mean():.3f} p50={np.median(values):.3f} "
            f"p95={np.percentile(values, 95):.3f} max={values.max():.3f}"
        )
    over = float((peak > args.error_threshold_m).mean())
    print(f"  sub-episodes with peak error > {args.error_threshold_m} m: {100 * over:.1f}%")
    print(f"labels: {total_labels - invalid_labels} valid, {invalid_labels} masked")
    num_lat = len(binding.JERK_LAT)
    print("label histogram (long jerk x lat jerk):")
    for action, count in sorted(label_counter.items()):
        long_idx, lat_idx = divmod(action, num_lat)
        print(
            f"  {action:3d}: long={binding.JERK_LONG[long_idx]:6.1f} lat={binding.JERK_LAT[lat_idx]:5.1f} "
            f"{100 * count / max(sum(label_counter.values()), 1):5.1f}%"
        )


if __name__ == "__main__":
    main()
