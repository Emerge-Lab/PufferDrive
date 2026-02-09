import copy
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

# Policy configurations: (filename, display_name, dynamics_model)
# Update these paths to match your actual checkpoint files
POLICY_CONFIGS = {
    "bc_policy": {
        "path": POLICY_DIR + "/bc_policy.pt",
        "dynamics": "classic",
        "type": "bc",
    },
    # "self_play_rl (classic)": {
    #     "path": POLICY_DIR + "/self_play_rl_simple_policy.pt",
    #     "dynamics": "classic",
    #     "type": "rl",
    # },
    # "guided_self_play_rl (classic)": {
    #     "path": POLICY_DIR + "/guided_self_play_classic_policy.pt",
    #     "dynamics": "classic",
    #     "type": "rl",
    # },
    # "self_play_rl (jerk)": {
    #     "path": POLICY_DIR + "/self_play_jerk_policy.pt",
    #     "dynamics": "jerk",
    #     "type": "rl",
    # },
    # "guided_self_play_rl (jerk)": {
    #     "path": POLICY_DIR + "/guided_self_play_jerk_policy.pt",
    #     "dynamics": "jerk",
    #     "type": "rl",
    # },
}

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
    """Evaluate an RL policy using WOSAC metrics."""

    # Use a copy to avoid mutating the original config
    config = copy.deepcopy(config)

    # Ensure evaluator is in RL mode (may have been changed by BC evaluation)
    evaluator.mode = "rl"

    # Enable RNN state initialization for LSTM-based policies
    config["train"]["use_rnn"] = True

    config["load_model_path"] = policy_path

    # Load policy
    policy = load_policy(config, vecenv, "puffer_drive")
    policy.eval()

    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

    # Roll out trained policy in the simulator
    simulated_trajectories = evaluator.collect_simulated_trajectories(
        args=config,
        puffer_env=vecenv,
        policy=policy,
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


def create_config_and_env(env_name, dynamics_model="classic"):
    """Create a config and vecenv for a specific dynamics model."""
    config = load_config(env_name)

    # Common WOSAC evaluation settings
    config["env"]["num_maps"] = 100
    config["env"]["map_dir"] = "pufferlib/resources/drive/binaries/validation"
    config["eval"]["wosac_target_scenarios"] = 1000
    config["eval"]["wosac_batch_size"] = 100
    config["eval"]["wosac_scenario_pool_size"] = 10_000
    config["wosac"]["enabled"] = True
    config["vec"]["backend"] = "PufferEnv"
    config["vec"]["num_envs"] = 1
    config["env"]["init_mode"] = "create_all_valid"
    config["env"]["control_mode"] = "control_wosac"
    config["env"]["init_steps"] = 10
    config["env"]["goal_behavior"] = 2
    config["env"]["goal_radius"] = 1.0
    config["env"]["save_data_to_disk"] = False

    # Set dynamics model
    config["env"]["dynamics_model"] = dynamics_model

    # Disable human data preparation for jerk dynamics (not implemented)
    if dynamics_model == "jerk":
        config["env"]["prep_human_data"] = False

    vecenv = load_env(env_name, config)
    evaluator = WOSACEvaluator(config)

    return config, vecenv, evaluator


def pipeline(env_name="puffer_drive"):
    """Obtain WOSAC scores for various baselines and policies across dynamics models."""

    all_results = []

    config_classic, vecenv_classic, evaluator_classic = create_config_and_env(env_name, "classic")

    # Ground truth (dynamics-agnostic, only need to run once)
    print("Evaluating: ground_truth")
    evaluator_classic.eval_mode = "ground_truth"
    df_results_gt = evaluator_classic.evaluate(config_classic, vecenv_classic, policy=None)
    df_results_gt["policy"] = "ground_truth"
    all_results.append(df_results_gt)

    # Inferred human actions (classic only - not implemented for jerk)
    print("Evaluating: inferred_human_actions (classic)")
    df_results_inferred_human = evaluate_human_inferred_actions(config_classic, vecenv_classic, evaluator_classic)
    df_results_inferred_human["policy"] = "inferred_human (classic)"
    all_results.append(df_results_inferred_human)

    # --- Classic dynamics evaluations ---
    print("=" * 60)
    print("Running evaluations with CLASSIC dynamics model...")
    print("=" * 60)

    # Random baseline for classic
    print("Evaluating: random (classic)")
    df_results_random_classic = evaluate_random_policy(config_classic, vecenv_classic, evaluator_classic)
    df_results_random_classic["policy"] = "random (classic)"
    all_results.append(df_results_random_classic)

    # Evaluate classic dynamics policies
    evaluator_classic.eval_mode = "policy"
    for policy_name, policy_cfg in POLICY_CONFIGS.items():
        if policy_cfg["dynamics"] != "classic":
            continue
        print(f"Evaluating: {policy_name}")
        if policy_cfg["type"] == "bc":
            df_result = evaluate_bc_policy(config_classic, vecenv_classic, evaluator_classic, policy_cfg["path"])
        else:
            df_result = evaluate_rl_policy(config_classic, vecenv_classic, evaluator_classic, policy_cfg["path"])
        df_result["policy"] = policy_name
        all_results.append(df_result)

    # --- Jerk dynamics evaluations ---
    # Check if any jerk policies are configured
    jerk_policies = {k: v for k, v in POLICY_CONFIGS.items() if v["dynamics"] == "jerk"}

    if jerk_policies:
        print("=" * 60)
        print("Running evaluations with JERK dynamics model...")
        print("=" * 60)

        config_jerk, vecenv_jerk, evaluator_jerk = create_config_and_env(env_name, "jerk")

        # Random baseline for jerk (different action space, so separate baseline)
        print("Evaluating: random (jerk)")
        df_results_random_jerk = evaluate_random_policy(config_jerk, vecenv_jerk, evaluator_jerk)
        df_results_random_jerk["policy"] = "random (jerk)"
        all_results.append(df_results_random_jerk)

        # Evaluate jerk dynamics policies
        for policy_name, policy_cfg in jerk_policies.items():
            print(f"Evaluating: {policy_name}")
            df_result = evaluate_rl_policy(config_jerk, vecenv_jerk, evaluator_jerk, policy_cfg["path"])
            df_result["policy"] = policy_name
            all_results.append(df_result)

    # Combine all results
    df = pd.concat(all_results, ignore_index=True)
    df = df[COLUMN_ORDER]

    # Visualize
    plot_wosac_results(df)
    plot_realism_score_distributions(df)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df.groupby("policy")["realism_meta_score"].mean().sort_values(ascending=False))
    print("---")
    print(df.groupby("policy")["kinematic_metrics"].mean())
    print("---")
    print(df.groupby("policy")["interactive_metrics"].mean())
    print("---")
    print(df.groupby("policy")["map_based_metrics"].mean())

if __name__ == "__main__":
    pipeline()
