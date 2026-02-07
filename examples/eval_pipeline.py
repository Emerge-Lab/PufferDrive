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
    "num_collisions_sim",
    "num_collisions_ref",
    "num_offroad_sim",
    "num_offroad_ref",
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

    total_scenes = df[df["policy"] == "random"]["realism_meta_score"].shape[0]

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

        ax.set_title(f"{policy} (n={total_scenes})")
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


def plot_sensitivity_to_collisions(
    collision_data_per_rollout, collision_data_per_timestep, ground_truth_results=None, random_policy_results=None
):
    """Create a 2-column visualization showing sensitivity to added collisions.

    Args:
        collision_data_per_rollout: List of tuples (num_collisions, results_df) for per-rollout collisions
        collision_data_per_timestep: List of tuples (num_collisions, results_df) for per-timestep collisions
        ground_truth_results: DataFrame with ground truth baseline results (optional)
        random_policy_results: DataFrame with random policy baseline results (optional)
    """

    # Set style
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract data for per-rollout collisions
    num_collisions_rollout = [x[0] for x in collision_data_per_rollout]
    meta_scores_rollout = [x[1]["realism_meta_score"].mean() for x in collision_data_per_rollout]
    meta_scores_std_rollout = [x[1]["realism_meta_score"].std() for x in collision_data_per_rollout]
    collision_likelihoods_rollout = [x[1]["likelihood_collision_indication"].mean() for x in collision_data_per_rollout]
    collision_likelihoods_std_rollout = [
        x[1]["likelihood_collision_indication"].std() for x in collision_data_per_rollout
    ]

    # Extract data for per-timestep collisions
    num_collisions_timestep = [x[0] for x in collision_data_per_timestep]
    meta_scores_timestep = [x[1]["realism_meta_score"].mean() for x in collision_data_per_timestep]
    meta_scores_std_timestep = [x[1]["realism_meta_score"].std() for x in collision_data_per_timestep]
    collision_likelihoods_timestep = [
        x[1]["likelihood_collision_indication"].mean() for x in collision_data_per_timestep
    ]
    collision_likelihoods_std_timestep = [
        x[1]["likelihood_collision_indication"].std() for x in collision_data_per_timestep
    ]

    # Left: Meta-score vs collisions
    axes[0].errorbar(
        num_collisions_rollout,
        meta_scores_rollout,
        yerr=meta_scores_std_rollout,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        label="Per-rollout",
        color="mediumpurple",
        linestyle="-",
        zorder=3,
    )
    axes[0].errorbar(
        num_collisions_timestep,
        meta_scores_timestep,
        yerr=meta_scores_std_timestep,
        marker="s",
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        label="Per-timestep",
        color="royalblue",
        linestyle="-",
        zorder=3,
    )

    # Add baseline reference lines
    if ground_truth_results is not None:
        gt_meta_score = ground_truth_results["realism_meta_score"].mean()
        axes[0].axhline(
            gt_meta_score, color="green", linestyle="--", linewidth=2, label="Ground truth", alpha=1.0, zorder=1
        )

    if random_policy_results is not None:
        random_meta_score = random_policy_results["realism_meta_score"].mean()
        axes[0].axhline(
            random_meta_score,
            color="tab:orange",
            linestyle="--",
            linewidth=2,
            label="Random policy",
            alpha=0.8,
            zorder=1,
        )

    axes[0].set_xlabel("Number of total added collisions across all rollouts")
    axes[0].set_ylabel("Realism meta score")
    axes[0].set_title("Sensitivity of meta-score to collisions")
    axes[0].grid(alpha=0.3, linestyle="--", zorder=0)
    axes[0].legend(facecolor="white")
    sns.despine(ax=axes[0])

    # Right: Collision likelihood vs collisions
    axes[1].errorbar(
        num_collisions_rollout,
        collision_likelihoods_rollout,
        yerr=collision_likelihoods_std_rollout,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        label="Per-rollout",
        color="mediumpurple",
        linestyle="-",
        zorder=3,
    )
    axes[1].errorbar(
        num_collisions_timestep,
        collision_likelihoods_timestep,
        yerr=collision_likelihoods_std_timestep,
        marker="s",
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        label="Per-timestep",
        color="royalblue",
        linestyle="-",
        zorder=3,
    )

    # Add baseline reference lines
    if ground_truth_results is not None:
        gt_collision_likelihood = ground_truth_results["likelihood_collision_indication"].mean()
        axes[1].axhline(
            gt_collision_likelihood,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Ground truth",
            alpha=1.0,
            zorder=1,
        )

    if random_policy_results is not None:
        random_collision_likelihood = random_policy_results["likelihood_collision_indication"].mean()
        axes[1].axhline(
            random_collision_likelihood,
            color="tab:orange",
            linestyle="--",
            linewidth=2,
            label="Random policy",
            alpha=0.8,
            zorder=1,
        )

    axes[1].set_xlabel("Number of added collisions across all rollouts")
    axes[1].set_ylabel("Collision likelihood score")
    axes[1].set_title("Sensitivity of collision likelihood to collisions")
    axes[1].grid(alpha=0.3, linestyle="--", zorder=0)
    axes[1].legend(facecolor="white")
    sns.despine(ax=axes[1])

    plt.tight_layout()
    plt.savefig("wosac_collision_sensitivity.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return fig


def plot_scenes_with_events(df, ref_collisions, ref_offroad):
    """Create plots showing collision and off-road events for ground truth reference data.

    Creates 1 figure with 2 subplots:
    1. Bar chart: Percentage of scenes with collision vs off-road events
    2. Histogram: Distribution of number of collisions/off-road events per agent

    Args:
        df: DataFrame with columns 'policy', 'num_collisions_ref', 'num_offroad_ref'
        ref_collisions: Array of shape (agents, 1, time) with collision indicators
        ref_offroad: Array of shape (agents, 1, time) with off-road indicators
    """

    # Filter to only ground truth
    gt_df = df[df["policy"] == "ground_truth"].copy()

    if len(gt_df) == 0:
        print("Warning: No ground_truth policy found in dataframe")
        return None

    # Check which columns are available
    has_collision_data = "num_collisions_ref" in gt_df.columns
    has_offroad_data = "num_offroad_ref" in gt_df.columns

    if not has_collision_data and not has_offroad_data:
        print("Warning: No collision or off-road data columns found")
        return None

    # Set style
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")

    total_scenes = len(gt_df)

    # Calculate counts
    scenes_with_collision = (gt_df["num_collisions_ref"] > 0).sum() if has_collision_data else 0
    scenes_with_offroad = (gt_df["num_offroad_ref"] > 0).sum() if has_offroad_data else 0

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Percentage bar chart (red colors for false positives)
    pct_collision = (scenes_with_collision / total_scenes) * 100
    pct_offroad = (scenes_with_offroad / total_scenes) * 100

    pct_data = pd.DataFrame({"Event type": ["Collision", "Off-road"], "Percentage": [pct_collision, pct_offroad]})

    sns.barplot(
        data=pct_data,
        x="Event type",
        y="Percentage",
        palette=["lightcoral", "darkred"],  # Red colors for false positives
        ax=axes[0],
        alpha=0.8,
    )

    axes[0].set_title("Scenes with at least one event (Reference)")
    axes[0].set_xlabel("Event type")
    axes[0].set_ylabel("Percentage of scenes (%)")
    axes[0].set_ylim(0, max(pct_collision, pct_offroad) * 1.15)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    # Add percentage labels on bars
    for i, (idx, row) in enumerate(pct_data.iterrows()):
        axes[0].text(
            i,
            row["Percentage"] + max(pct_collision, pct_offroad) * 0.02,
            f"{row['Percentage']:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add total scenes
    axes[0].text(
        0.98,
        0.98,
        f"Total scenes: {total_scenes}",
        transform=axes[0].transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
    )

    sns.despine(ax=axes[0])

    # Right plot: Distribution histogram
    # Calculate number of collisions and off-road events per agent across all timesteps
    collisions_per_agent = ref_collisions.sum(axis=(1, 2))  # Sum across rollout and time dimensions
    offroad_per_agent = ref_offroad.sum(axis=(1, 2))  # Sum across rollout and time dimensions

    # Create histogram data
    hist_data = []
    for count in collisions_per_agent:
        hist_data.append({"Event type": "Collision", "Count": count})
    for count in offroad_per_agent:
        hist_data.append({"Event type": "Off-road", "Count": count})

    hist_df = pd.DataFrame(hist_data)

    # Plot overlapping histograms
    colors = {"Collision": "lightcoral", "Off-road": "darkred"}  # Red colors for false positives
    for event_type in ["Collision", "Off-road"]:
        data = hist_df[hist_df["Event type"] == event_type]["Count"]
        axes[1].hist(data, bins=20, alpha=0.6, color=colors[event_type], edgecolor="black", label=event_type)

    axes[1].set_title("Distribution of events per agent")
    axes[1].set_xlabel("Number of events per agent")
    axes[1].set_ylabel("Number of agents")
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].legend()

    # Add statistics
    total_agents = len(collisions_per_agent)
    total_collisions = collisions_per_agent.sum()
    total_offroad = offroad_per_agent.sum()
    agents_with_collisions = (collisions_per_agent > 0).sum()
    agents_with_offroad = (offroad_per_agent > 0).sum()
    pct_agents_collision = (agents_with_collisions / total_agents) * 100
    pct_agents_offroad = (agents_with_offroad / total_agents) * 100

    stats_text = f"Total agents: {total_agents}\n"
    stats_text += f"\nCollision:\n"
    stats_text += f"  Total events: {total_collisions}\n"
    stats_text += f"  Agents w/ ≥1: {pct_agents_collision:.1f}%\n"
    stats_text += f"\nOff-road:\n"
    stats_text += f"  Total events: {total_offroad}\n"
    stats_text += f"  Agents w/ ≥1: {pct_agents_offroad:.1f}%"

    axes[1].text(
        0.98,
        0.98,
        stats_text,
        transform=axes[1].transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
    )

    sns.despine(ax=axes[1])

    plt.tight_layout()
    plt.savefig("wosac_scenes_with_events.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return fig


def evaluate_ground_truth(config, vecenv, evaluator):
    """Compute WOSAC metrics for ground truth trajectories."""

    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

    fake_simulated_trajectories = gt_trajectories.copy()
    for key in ["x", "y", "heading", "id"]:
        fake_simulated_trajectories[key] = np.repeat(gt_trajectories[key], config["eval"]["wosac_num_rollouts"], axis=1)
    fake_simulated_trajectories["id"] = fake_simulated_trajectories["id"][..., np.newaxis]
    fake_simulated_trajectories["dones"] = np.zeros_like(fake_simulated_trajectories["x"])

    # Compute metrics
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
    results = evaluator.compute_metrics(
        gt_trajectories,
        fake_simulated_trajectories,
        agent_state,
        road_edge_polylines,
    )
    return results, evaluator.ref_collisions, evaluator.ref_offroad


def collision_sweep_from_ground_truth(
    config, vecenv, evaluator, num_collisions_per_rollout=0, num_collisions_per_timestep=0
):
    """Compute WOSAC metrics for ground truth trajectories with added collisions.

    Args:
        config: Configuration dictionary
        vecenv: PufferLib environment
        evaluator: WOSACEvaluator instance
        num_collisions_per_rollout: Number of collisions to add per rollout (added to first timestep)
        num_collisions_per_timestep: Number of collisions to add per timestep (spread across time)
    """

    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

    fake_simulated_trajectories = gt_trajectories.copy()
    for key in ["x", "y", "heading", "id"]:
        fake_simulated_trajectories[key] = np.repeat(gt_trajectories[key], config["eval"]["wosac_num_rollouts"], axis=1)
    fake_simulated_trajectories["id"] = fake_simulated_trajectories["id"][..., np.newaxis]
    fake_simulated_trajectories["dones"] = np.zeros_like(fake_simulated_trajectories["x"])

    # Compute metrics
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
    results = evaluator.compute_metrics(
        gt_trajectories,
        fake_simulated_trajectories,
        agent_state,
        road_edge_polylines,
        collisions_to_add_per_rollout=num_collisions_per_rollout,
        collisions_to_add_per_timestep=num_collisions_per_timestep,
    )
    return results


def run_collision_sensitivity_sweep(config, vecenv, evaluator, collision_counts):
    """Run collision sensitivity analysis for both per-rollout and per-timestep collisions.

    Args:
        config: Configuration dictionary
        vecenv: PufferLib environment
        evaluator: WOSACEvaluator instance
        collision_counts: List of collision counts to test (e.g., [0, 5, 10, 15, 20])

    Returns:
        Tuple of (per_rollout_data, per_timestep_data) where each is a list of (count, results_df) tuples
    """
    per_rollout_data = []
    per_timestep_data = []

    print("Running per-rollout collision sweep...")
    for num_colls in collision_counts:
        print(f"  Testing {num_colls} collisions per rollout...")
        results = collision_sweep_from_ground_truth(
            config, vecenv, evaluator, num_collisions_per_rollout=num_colls, num_collisions_per_timestep=0
        )
        per_rollout_data.append((num_colls, results))

    print("\nRunning per-timestep collision sweep...")
    for num_colls in collision_counts:
        print(f"  Testing {num_colls} collisions per timestep...")
        results = collision_sweep_from_ground_truth(
            config, vecenv, evaluator, num_collisions_per_rollout=0, num_collisions_per_timestep=num_colls
        )
        per_timestep_data.append((num_colls, results))

    return per_rollout_data, per_timestep_data


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

    # Dataset configuration
    config["env"]["map_dir"] = "pufferlib/resources/drive/binaries/training"
    config["eval"]["wosac_target_scenarios"] = 1000
    config["eval"]["wosac_batch_size"] = 100
    config["eval"]["wosac_scenario_pool_size"] = 10_0000

    # WOSAC settings
    config["wosac"]["enabled"] = True
    config["vec"]["backend"] = "PufferEnv"
    config["vec"]["num_envs"] = 1
    config["env"]["init_mode"] = "create_all_valid"
    config["env"]["control_mode"] = "control_wosac"
    config["env"]["init_steps"] = 10
    config["env"]["goal_behavior"] = 2  # Stop at goal
    config["env"]["goal_radius"] = 1.0
    config["env"]["save_data_to_disk"] = False

    # Make env
    vecenv = load_env(env_name, config)

    # Make evaluator
    evaluator = WOSACEvaluator(config)

    # Baseline: Ground truth
    evaluator.eval_mode = "ground_truth"
    df_results_gt = evaluator.evaluate(config, vecenv, policy=None)
    ref_collisions, ref_offroad = evaluator.ref_collisions, evaluator.ref_offroad
    df_results_gt["policy"] = "ground_truth"

    # Baseline: Random policy
    # df_results_random = evaluate_random_policy(config, vecenv, evaluator)
    # df_results_random["policy"] = "random"

    # Combine results for basic plots
    df = pd.concat(
        [
            df_results_gt,
            # df_results_random,
        ],
        ignore_index=True,
    )

    df = df[COLUMN_ORDER]

    # Visualize basic results
    plot_wosac_results(df)
    plot_realism_score_distributions(df)
    plot_scenes_with_events(df, ref_collisions, ref_offroad)

    # # Run collision sensitivity analysis
    # collision_counts = [0, 1, 2, 4, 8, 16, 32]
    # per_rollout_data, per_timestep_data = run_collision_sensitivity_sweep(
    #     config, vecenv, evaluator, collision_counts
    # )

    # # Plot collision sensitivity with baseline references
    # plot_sensitivity_to_collisions(per_rollout_data, per_timestep_data,
    #                                ground_truth_results=df_results_gt,
    #                                random_policy_results=df_results_random)

    print(f"total agents: {df_results_gt['num_agents_per_scene'].sum().item()}")

    print(df.groupby("policy")["realism_meta_score"].mean())
    print("---")
    print(df.groupby("policy")["kinematic_metrics"].mean())
    print("---")
    print(df.groupby("policy")["interactive_metrics"].mean())
    print("---")
    print(df.groupby("policy")["map_based_metrics"].mean())


if __name__ == "__main__":
    pipeline()
