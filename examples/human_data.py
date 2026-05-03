"""Evaluate inferred-expert-action quality on the delta-local dynamics model."""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pufferlib.pufferl import load_config, load_env
from pufferlib.ocean.benchmark.evaluator_minimal import CheckpointEvaluator


ENV_NAME = "puffer_drive"
NUM_ITERATIONS = 10
EPISODE_LENGTH = 91
NUM_VIDEOS = 1
VIDEO_DIR = "videos/inferred_expert"
PLOT_DIR = "plots/inferred_expert"


def main():
    args = load_config(ENV_NAME)
    args["env"]["control_mode"] = "inferred_expert_actions"
    args["env"]["episode_length"] = EPISODE_LENGTH
    args["env"]["termination_mode"] = 1
    args["env"]["num_agents"] = 256
    args["env"]["async_resets"] = False
    args["env"]["resample_frequency"] = 0
    args["vec"] = dict(backend="PufferEnv", num_envs=1)

    env = load_env(ENV_NAME, args)
    evaluator = CheckpointEvaluator(args)

    num_envs = env.driver_env.num_envs
    total_scenarios = NUM_ITERATIONS * num_envs
    print(f"\nTotal scenarios: {total_scenarios} ({NUM_ITERATIONS} iterations x {num_envs} envs)")

    rows = []
    for it in range(NUM_ITERATIONS):
        env_logs = evaluator.rollout(env=env, policy=None, render_env_idx=it)
        populated = [log for log in env_logs if log and log.get("n", 0) > 0]

        for log in populated:
            rows.append(
                {
                    "iteration": it,
                    "route_progress": log["route_progress"],
                    "lateral_error": log["lateral_error_avg"],
                    "longitudinal_error": log["longitudinal_error_avg"],
                    "collision_rate": log["collision_rate"],
                    "offroad_rate": log["offroad_rate"],
                }
            )

    df = pd.DataFrame(rows)

    if len(df):
        print(f"\nPooled samples across {NUM_ITERATIONS} iterations: {len(df)}")
        print("\nMean ± std:")
        rate_metrics = {"collision_rate", "offroad_rate", "route_progress"}
        for k in ["collision_rate", "offroad_rate", "route_progress", "lateral_error", "longitudinal_error"]:
            if k in rate_metrics:
                print(f"  {k:<20s} {df[k].mean() * 100:.2f}% ± {df[k].std() * 100:.2f}%")
            else:
                print(f"  {k:<20s} {df[k].mean():.4f} ± {df[k].std():.4f}")

    # Plots
    os.makedirs(PLOT_DIR, exist_ok=True)

    rate_cols = ["collision_rate", "offroad_rate", "route_progress"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100)

    rate_means = df[rate_cols].mean().values * 100
    rate_stds = df[rate_cols].std().values * 100
    axes[0].bar(rate_cols, rate_means, yerr=rate_stds, capsize=4, color=["tab:red", "tab:orange", "tab:green"])
    axes[0].set_ylabel("Perc. (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title(f"Inferred expert SDC rates (n={len(df)})")
    for i, (m, s) in enumerate(zip(rate_means, rate_stds)):
        axes[0].text(i, m + s + 2, f"{m:.1f}%", ha="center", va="bottom", fontsize=10)

    axes[1].hist(df["lateral_error"], bins=40, color="tab:blue", edgecolor="black")
    axes[1].set_xlabel("lateral error (m)")
    axes[1].set_title(f"Lateral error  mean={df['lateral_error'].mean():.2f}m")

    axes[2].hist(df["longitudinal_error"], bins=40, color="tab:purple", edgecolor="black")
    axes[2].set_xlabel("longitudinal error (m)")
    axes[2].set_title(f"Longitudinal error  mean={df['longitudinal_error'].mean():.2f}m")

    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/summary.png", dpi=120)
    plt.close(fig)

    print(f"\nPlots saved to {PLOT_DIR}/")

    # Render a video per env up to NUM_VIDEOS
    os.makedirs(VIDEO_DIR, exist_ok=True)
    n_videos = min(NUM_VIDEOS, num_envs)
    print(f"\nRecording {n_videos} videos to {VIDEO_DIR}/")
    for env_idx in range(n_videos):
        evaluator.rollout(env=env, policy=None, render_env_idx=env_idx)
        env.driver_env.stop_recorder(env_idx)
        scenario_id = env.driver_env.scenario_ids[env_idx]
        src = f"{scenario_id}.mp4"
        if os.path.exists(src):
            dst = os.path.join(VIDEO_DIR, src)
            shutil.move(src, dst)
            print(f"  env {env_idx}: {dst}")
        else:
            print(f"  env {env_idx}: missing {src}")

    env.close()


if __name__ == "__main__":
    main()
