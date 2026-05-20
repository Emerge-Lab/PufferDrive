#!/usr/bin/env python3
"""Mine policy failures by direct Drive() rollout.

Bypasses pufferl.mine_failures (which forks via pufferlib.vector and hangs on
CUDA-after-fork). Uses Drive's built-in vectorization: control_sdc_only +
num_agents=N gives N parallel SDC scenarios in one Drive instance, one process.

C-side compact-replay capture (capture_compact_replay=True) attaches a bundle
to each completed_episode info dict; we save those directly and render HTML
via pufferlib.mining_viz.

Usage:
    python scripts/mine_failures.py \\
        --checkpoint /scratch/$USER/.../model_puffer_drive_011000.pt \\
        --map-dir /scratch/$USER/data/nuplan/nuplan_mini_train_bins \\
        --output-dir /scratch/$USER/failure_mining/out \\
        --num-scenarios 30 --num-maps 8 --num-agents 16
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from pufferlib import mining_viz
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.torch import Drive as DrivePolicy


def build_env(args):
    return Drive(
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        num_agents=args.num_agents,
        max_agents_per_env=1,
        min_agents_per_env=1,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        init_steps=args.init_steps,
        scenario_length=args.scenario_length,
        goal_radius=args.goal_radius,
        reward_randomization=False,
        # Prevent the auto vec_log from draining env->log mid-rollout; we want
        # completed_episode summaries to land in info as the episodes finish.
        report_interval=args.scenario_length * 10,
        emit_completed_episodes=True,
        capture_compact_replay=True,
        dynamics_model=args.dynamics_model,
        num_target_waypoints=args.num_target_waypoints,
        max_partner_observations=args.max_partner_observations,
        max_lane_segment_observations=args.max_lane_segment_observations,
        max_boundary_segment_observations=args.max_boundary_segment_observations,
        max_traffic_control_observations=args.max_traffic_control_observations,
        resample_frequency=0,
        eval_mode=1,
        seed=args.seed,
    )


def build_policy(env, args, device):
    policy = DrivePolicy(
        env=env,
        input_size=args.input_size,
        backbone_hidden_size=args.backbone_hidden_size,
        backbone_num_layers=args.backbone_num_layers,
        actor_hidden_size=args.actor_hidden_size,
        actor_num_layers=args.actor_num_layers,
        critic_hidden_size=args.critic_hidden_size,
        critic_num_layers=args.critic_num_layers,
        encoder_gigaflow=args.encoder_gigaflow,
        dropout=0.0,
        split_network=args.split_network,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    policy.load_state_dict(state)
    return policy.eval().to(device)


def sample_actions(policy, obs, device, deterministic=True):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_tuple, _ = policy(obs_t)
        if deterministic:
            actions = torch.stack([logit.argmax(dim=-1) for logit in logits_tuple], dim=-1)
        else:
            actions = torch.stack(
                [torch.distributions.Categorical(logits=logit).sample() for logit in logits_tuple],
                dim=-1,
            )
    return actions.cpu().numpy().astype(np.int64)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mine] device={device}", flush=True)

    replays_dir = os.path.join(args.output_dir, "replays")
    renders_dir = os.path.join(args.output_dir, "renders") if args.render else None
    os.makedirs(replays_dir, exist_ok=True)
    if renders_dir is not None:
        os.makedirs(renders_dir, exist_ok=True)

    rows = []
    next_episode_id = 0
    t_start = time.time()

    while next_episode_id < args.num_scenarios:
        env = build_env(args)
        n_envs = env.num_envs
        if next_episode_id == 0:
            policy = build_policy(env, args, device)
            print(
                f"[mine] obs_shape={env.observations.shape} num_envs={n_envs} "
                f"ego_features={env.ego_features}",
                flush=True,
            )

        obs, _ = env.reset(seed=args.seed + next_episode_id)
        completed_summaries = []
        for _ in range(args.scenario_length):
            actions = sample_actions(policy, obs, device, deterministic=args.deterministic)
            obs, _, _, _, info = env.step(actions)
            for entry in info:
                if isinstance(entry, dict) and entry.get("summary_type") == "completed_episode":
                    completed_summaries.append(entry)

        for summary in completed_summaries:
            if next_episode_id >= args.num_scenarios:
                break
            bundle = summary.pop("compact_replay_bundle", None)
            row = {k: v for k, v in summary.items()}
            row["episode_id"] = next_episode_id
            total_dist = float(row.get("total_distance_travelled", 0.0))
            total_infr = max(1.0, float(row.get("total_infractions", 0.0)))
            row["avg_distance_per_infraction"] = total_dist / total_infr
            row["has_replay"] = 0
            row["replay_path"] = None
            if bundle is not None:
                replay_path = os.path.join(replays_dir, f"episode_{next_episode_id:06d}.replay.zlib")
                with open(replay_path, "wb") as f:
                    f.write(bundle)
                row["has_replay"] = 1
                row["replay_path"] = replay_path
            rows.append(row)
            next_episode_id += 1

        elapsed = time.time() - t_start
        print(
            f"[mine] {next_episode_id}/{args.num_scenarios} captured "
            f"({elapsed:.1f}s, {next_episode_id / max(elapsed, 1e-6):.2f} eps/s)",
            flush=True,
        )

    episodes_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "episodes.csv")
    episodes_df.to_csv(csv_path, index=False)
    print(f"[mine] wrote {csv_path} ({len(rows)} episodes, {int(episodes_df['has_replay'].sum())} bundles)")

    if args.render and renders_dir is not None:
        render_lookup = {}
        rendered = 0
        for row in rows:
            if not row.get("has_replay"):
                continue
            ep_id = int(row["episode_id"])
            out_html = os.path.join(renders_dir, f"episode_{ep_id:06d}.html")
            mining_viz.render_compact_replay_html(row["replay_path"], out_html, render_context={"summary": row})
            render_lookup[ep_id] = os.path.relpath(out_html, renders_dir)
            rendered += 1
        index_path = os.path.join(renders_dir, "index.html")
        mining_viz.generate_failure_index(episodes_df, render_lookup, index_path)
        print(f"[mine] rendered {rendered} replays + index at {index_path}")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        numeric = episodes_df.select_dtypes(include=[np.number]).mean(numeric_only=True).to_dict()
        json.dump(
            {
                "num_episodes": int(len(episodes_df)),
                "num_bundles": int(episodes_df["has_replay"].sum()) if "has_replay" in episodes_df else 0,
                "elapsed_seconds": time.time() - t_start,
                "metrics_mean": {k: float(v) for k, v in numeric.items()},
            },
            f,
            indent=2,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--map-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-scenarios", type=int, default=30)
    p.add_argument("--num-maps", type=int, default=8)
    p.add_argument(
        "--num-agents",
        type=int,
        default=16,
        help="parallel SDC scenarios per Drive env build (= env.num_envs in control_sdc_only)",
    )
    p.add_argument("--scenario-length", type=int, default=91)
    p.add_argument(
        "--goal-radius",
        type=float,
        default=6.0,
        help="xy distance (m) at which a goal counts as reached; z gate is fixed at Z_BUFFER=4m",
    )
    p.add_argument("--init-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--render", action="store_true", default=True)
    # Env obs config (must match what the policy was trained with)
    p.add_argument("--dynamics-model", default="jerk")
    p.add_argument("--num-target-waypoints", type=int, default=3)
    p.add_argument("--max-partner-observations", type=int, default=16)
    p.add_argument("--max-lane-segment-observations", type=int, default=80)
    p.add_argument("--max-boundary-segment-observations", type=int, default=80)
    p.add_argument("--max-traffic-control-observations", type=int, default=4)
    # Policy config (must match the training-time policy)
    p.add_argument("--input-size", type=int, default=128)
    p.add_argument("--backbone-hidden-size", type=int, default=512)
    p.add_argument("--backbone-num-layers", type=int, default=4)
    p.add_argument("--actor-hidden-size", type=int, default=512)
    p.add_argument("--actor-num-layers", type=int, default=0)
    p.add_argument("--critic-hidden-size", type=int, default=512)
    p.add_argument("--critic-num-layers", type=int, default=0)
    p.add_argument("--encoder-gigaflow", action="store_true", default=True)
    p.add_argument("--split-network", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    main()
