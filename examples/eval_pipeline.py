import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings

from pufferlib.pufferl import load_env, load_policy, load_config
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

POLICY_DIR = "models"

COLUMN_ORDER = [
    "policy",
    "realism_meta_score",
    "kinematic_metrics",
    "interactive_metrics",
    "map_based_metrics",
    "min_ade",
    "ade",
    "likelihood_linear_speed",
    "likelihood_linear_acceleration",
    "likelihood_angular_speed",
    "likelihood_angular_acceleration",
    "likelihood_collision_indication",
    "likelihood_offroad_indication",
    "likelihood_time_to_collision",
    "likelihood_distance_to_road_edge",
    "likelihood_distance_to_nearest_object",
]


def plot_wosac_results(df):
    """Create a 3-column visualization of WOSAC results."""

    # Set style
    sns.set("notebook", font_scale=1.05, rc={"figure.figsize": (16, 5)})
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")
    mpl.rcParams["lines.markersize"] = 8

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    palette = sns.color_palette("Set2", n_colors=len(df["policy"].unique()))

    # Left: Realism meta score
    sns.barplot(data=df, x="policy", y="realism_meta_score", errorbar="sd", palette=palette, ax=axes[0], alpha=0.8)
    axes[0].set_title("Aggregate realism score")
    # axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    axes[0].tick_params(axis="x", rotation=30)

    # Middle: Metric Categories
    df_metrics = df.melt(
        id_vars=["policy"], value_vars=["kinematic_metrics", "interactive_metrics", "map_based_metrics"]
    )
    sns.barplot(
        data=df_metrics, x="variable", y="value", hue="policy", errorbar="sd", palette=palette, ax=axes[1], alpha=0.8
    )
    axes[1].set_title("Group metric categories")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Score")
    # axes[1].set_ylim(0, 1.0)
    axes[1].legend(title="Policy", loc="upper left")
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].tick_params(axis="x", rotation=30)

    # Right: ADE and minADE
    df_ade = df.melt(id_vars=["policy"], value_vars=["ade", "min_ade"])
    sns.barplot(
        data=df_ade, x="policy", y="value", hue="variable", errorbar="sd", palette="muted", ax=axes[2], alpha=0.8
    )
    axes[2].set_title("Displacement error")
    axes[2].set_ylabel("Distance (m)")
    axes[2].legend(title="Metric")
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")
    axes[2].tick_params(axis="x", rotation=15)

    for ax in axes:
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig("wosac_evaluation_results.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return fig


def plot_realism_score_distributions(df):
    """Create histogram distributions of realism scores for each policy."""

    # Set style
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")
    mpl.rcParams["lines.markersize"] = 8

    policies = df["policy"].unique()
    n_policies = len(policies)

    fig, axes = plt.subplots(1, n_policies, figsize=(5 * n_policies, 4), sharey=True, sharex=True)
    if n_policies == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=n_policies)

    for idx, (policy, ax) in enumerate(zip(policies, axes)):
        policy_data = df[df["policy"] == policy]["realism_meta_score"]

        # Plot histogram
        ax.hist(policy_data, bins=20, alpha=0.8, color=palette[idx], edgecolor="black")

        # Add mean line
        mean_val = policy_data.mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.3f}")

        # Add std text
        std_val = policy_data.std()
        ax.text(
            0.05,
            0.95,
            f"μ = {mean_val:.3f}\nσ = {std_val:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_title(f"{policy}")
        ax.set_xlabel("Realism meta score")
        if idx == 0:
            ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend()
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig("wosac_realism_score_distributions.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return fig


def evaluate_ground_truth(config, vecenv, evaluator):
    """Compute WOSAC metrics for ground truth trajectories."""

    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

    fake_simulated_trajectories = gt_trajectories.copy()
    for key in ["x", "y", "heading", "id"]:
        fake_simulated_trajectories[key] = np.repeat(gt_trajectories[key], config["eval"]["wosac_num_rollouts"], axis=1)
    fake_simulated_trajectories["id"] = fake_simulated_trajectories["id"][..., np.newaxis]

    # Compute metrics
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
    results = evaluator.compute_metrics(
        gt_trajectories,
        fake_simulated_trajectories,
        agent_state,
        road_edge_polylines,
    )
    return results


def evaluate_human_inferred_actions(config, vecenv, evaluator):
    """Compute WOSAC metrics for human inferred actions."""

    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

    vecenv._prep_human_data()

    # Roll out inferred human actions in the simulator
    simulated_trajectories = evaluator.collect_simulated_trajectories(
        args=config,
        puffer_env=vecenv,
        policy=None,
        actions=vecenv.expert_actions_discrete,
    )

    # Compute metrics
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
    results = evaluator.compute_metrics(
        gt_trajectories,
        simulated_trajectories,
        agent_state,
        road_edge_polylines,
    )

    return results


def evaluate_random_policy(config, vecenv, evaluator):
    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)
    simulated_trajectories = evaluator.collect_wosac_random_baseline(vecenv)

    # Compute metrics
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
    results = evaluator.compute_metrics(
        gt_trajectories,
        simulated_trajectories,
        agent_state,
        road_edge_polylines,
    )

    return results


def evaluate_bc_policy(config, vecenv, evaluator, policy_path):
    config["train"]["use_rnn"] = False
    evaluator.mode = "bc_policy"

    from train_bc_policy import BCPolicy

    bc_policy = BCPolicy(
        input_size=vecenv.observation_space.shape[-1],
        hidden_size=1024,
        output_size=21 * 31,
    )
    bc_policy.load_state_dict(torch.load(policy_path))
    bc_policy.eval().to(config["train"]["device"])

    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)
    simulated_trajectories = evaluator.collect_simulated_trajectories(
        args=config,
        puffer_env=vecenv,
        policy=bc_policy,
    )

    # Compute metrics
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
    results = evaluator.compute_metrics(
        gt_trajectories,
        simulated_trajectories,
        agent_state,
        road_edge_polylines,
    )

    return results


def evaluate_rl_policy(config, vecenv, evaluator, policy_path):
    config["train"]["use_rnn"] = True
    evaluator.mode = "rl_policy"

    policy = load_policy(config, vecenv, "puffer_drive")

    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)
    simulated_trajectories = evaluator.collect_simulated_trajectories(config, vecenv, policy)

    # Compute metrics
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
    results = evaluator.compute_metrics(
        gt_trajectories,
        simulated_trajectories,
        agent_state,
        road_edge_polylines,
    )
    return results


def pipeline(env_name="puffer_drive"):
    """Obtain WOSAC scores for various baselines and policies."""

    config = load_config(env_name)

    config["env"]["num_maps"] = 229
    config["env"]["map_dir"] = "pufferlib/resources/drive/binaries/validation_interactive_small"
    config["wosac"]["enabled"] = True
    config["vec"]["backend"] = "PufferEnv"
    config["vec"]["num_envs"] = 1
    config["env"]["sequential_map_sampling"] = True
    config["env"]["init_mode"] = "create_all_valid"
    config["env"]["control_mode"] = "control_wosac"
    config["env"]["init_steps"] = 10
    config["env"]["goal_behavior"] = 2
    config["env"]["goal_radius"] = 1.0

    # Make env
    vecenv = load_env(env_name, config)

    # Make evaluator
    evaluator = WOSACEvaluator(config)

    # Baseline: Ground truth
    df_results_gt = evaluate_ground_truth(config, vecenv, evaluator)
    df_results_gt["policy"] = "ground_truth"

    # Baseline: Agent with inferred human actions (using classic bicycle dynamics model)
    df_results_inferred_human = evaluate_human_inferred_actions(config, vecenv, evaluator)
    df_results_inferred_human["policy"] = "inferred_human_actions"

    # Baseline: Imitation learning policy
    # df_results_bc = evaluate_bc_policy(config, vecenv, evaluator, POLICY_DIR + "/bc_policy.pt")
    # df_results_bc["policy"] = "bc_policy"

    # Baseline: Self-play RL policy
    # run: https://wandb.ai/emerge_/gsp/runs/qld2z6tn?nw=nwuserdaphnecor
    df_results_self_play = evaluate_rl_policy(
        config, vecenv, evaluator, "pufferlib/resources/drive/pufferdrive_weights.pt"
    )  # POLICY_DIR + "/puffer_drive_sp_qld2z6tn.pt")
    df_results_self_play["policy"] = "self_play_rl_base"

    # TODO: Guided self-play policy (guidance in rewards)
    # ...

    # TODO: Guided self-play policy (regularization)
    # ...

    # Baseline: Random policy
    df_results_random = evaluate_random_policy(config, vecenv, evaluator)
    df_results_random["policy"] = "random"

    # Combine
    df = pd.concat(
        [
            df_results_gt,
            df_results_inferred_human,
            df_results_random,
            # df_results_bc,
            df_results_self_play,
        ],
        ignore_index=True,
    )

    df = df[COLUMN_ORDER]

    # Visualize
    plot_wosac_results(df)
    plot_realism_score_distributions(df)

    print(df.groupby("policy")["realism_meta_score"].mean())
    print(df.shape[0])


if __name__ == "__main__":
    pipeline()
