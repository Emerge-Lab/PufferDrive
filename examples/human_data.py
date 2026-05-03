"""Evaluate inferred-expert-action quality on the delta-local dynamics model.

Compares discrete (bin-quantized) vs continuous (direct float) expert actions.
"""

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
NUM_VIDEOS = 5
ACTION_TYPES = ["discrete", "continuous"]
VIDEO_DIR = "videos/inferred_expert"
PLOT_DIR = "plots/inferred_expert"


def run_eval(action_type):
    args = load_config(ENV_NAME)
    args["env"]["control_mode"] = "inferred_expert_actions"
    args["env"]["action_type"] = action_type
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
    print(f"\n[{action_type}] Total scenarios: {total_scenarios} ({NUM_ITERATIONS} iterations x {num_envs} envs)")

    rows = []
    for it in range(NUM_ITERATIONS):
        env_logs = evaluator.rollout(env=env, policy=None, render_env_idx=0)
        populated = [log for log in env_logs if log and log.get("n", 0) > 0]

        for log in populated:
            rows.append(
                {
                    "action_type": action_type,
                    "iteration": it,
                    "route_progress": log["route_progress"],
                    "lateral_error": log["lateral_error_avg"],
                    "longitudinal_error": log["longitudinal_error_avg"],
                    "collision_rate": log["collision_rate"],
                    "offroad_rate": log["offroad_rate"],
                }
            )

    # Render videos
    os.makedirs(f"{VIDEO_DIR}/{action_type}", exist_ok=True)
    n_videos = min(NUM_VIDEOS, num_envs)
    print(f"[{action_type}] Recording {n_videos} videos to {VIDEO_DIR}/{action_type}/")
    for env_idx in range(n_videos):
        evaluator.rollout(env=env, policy=None, render_env_idx=env_idx)
        env.driver_env.stop_recorder(env_idx)
        scenario_id = env.driver_env.scenario_ids[env_idx]
        src = f"{scenario_id}.mp4"
        if os.path.exists(src):
            dst = os.path.join(f"{VIDEO_DIR}/{action_type}", src)
            shutil.move(src, dst)
            print(f"  env {env_idx}: {dst}")
        else:
            print(f"  env {env_idx}: missing {src}")

    env.close()
    return pd.DataFrame(rows)


def main():
    df = pd.concat([run_eval(at) for at in ACTION_TYPES], ignore_index=True)

    print(f"\nTotal pooled samples: {len(df)}")
    print("\nMean ± std by action_type:")
    rate_metrics = {"collision_rate", "offroad_rate", "route_progress"}
    for at in ACTION_TYPES:
        sub = df[df["action_type"] == at]
        print(f"\n  [{at}] n={len(sub)}")
        for k in ["collision_rate", "offroad_rate", "route_progress", "lateral_error", "longitudinal_error"]:
            if k in rate_metrics:
                print(f"    {k:<20s} {sub[k].mean() * 100:.2f}% ± {sub[k].std() * 100:.2f}%")
            else:
                print(f"    {k:<20s} {sub[k].mean():.4f} ± {sub[k].std():.4f}")

    os.makedirs(PLOT_DIR, exist_ok=True)

    rate_cols = ["collision_rate", "offroad_rate", "route_progress"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Grouped bar: rates per action_type
    x = np.arange(len(rate_cols))
    width = 0.38
    palette = {"discrete": "tab:blue", "continuous": "tab:purple"}
    for i, at in enumerate(ACTION_TYPES):
        sub = df[df["action_type"] == at]
        means = sub[rate_cols].mean().values * 100
        stds = sub[rate_cols].std().values * 100
        offset = (i - 0.5) * width
        bars = axes[0].bar(x + offset, means, width, yerr=stds, capsize=3, color=palette[at], label=at)
        for j, (m, s) in enumerate(zip(means, stds)):
            axes[0].text(x[j] + offset, m + s + 2, f"{m:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rate_cols)
    axes[0].set_ylabel("percent (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title(f"SDC rates (n={len(df)})")
    axes[0].legend(frameon=False)

    # Overlaid histograms
    for at in ACTION_TYPES:
        sub = df[df["action_type"] == at]
        axes[1].hist(sub["lateral_error"], bins=40, alpha=0.55, color=palette[at], edgecolor="black", label=at)
        axes[2].hist(sub["longitudinal_error"], bins=40, alpha=0.55, color=palette[at], edgecolor="black", label=at)

    axes[1].set_xlabel("lateral error (m)")
    axes[1].set_title("Lateral error")
    axes[1].legend(frameon=False)
    axes[2].set_xlabel("longitudinal error (m)")
    axes[2].set_title("Longitudinal error")
    axes[2].legend(frameon=False)

    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/summary.png", dpi=120)
    plt.close(fig)

    print(f"\nPlots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
