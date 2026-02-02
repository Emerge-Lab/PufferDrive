import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
import os

from pufferlib.pufferl import load_env, load_policy, load_config
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator
from pufferlib.ocean.benchmark import metrics

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


def collision_sweep_from_ground_truth(
    config,
    vecenv,
    evaluator,
    max_k=10,
    sweep_collision_rollouts=False,
):
    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)
    num_rollouts = config["eval"]["wosac_num_rollouts"]
    fake_simulated_trajectories = gt_trajectories.copy()
    for key in ["x", "y", "heading", "id"]:
        fake_simulated_trajectories[key] = np.repeat(gt_trajectories[key], num_rollouts, axis=1)
    fake_simulated_trajectories["id"] = fake_simulated_trajectories["id"][..., np.newaxis]
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()

    eval_mask = gt_trajectories["id"][:, 0] >= 0
    valid_mask = np.repeat(gt_trajectories["valid"][eval_mask].astype(bool), num_rollouts, axis=1)

    _, base_collision_per_step, _ = metrics.compute_interaction_features(
        fake_simulated_trajectories["x"],
        fake_simulated_trajectories["y"],
        fake_simulated_trajectories["heading"],
        gt_trajectories["scenario_id"],
        agent_state["length"],
        agent_state["width"],
        eval_mask,
        device=config["train"]["device"],
    )

    if sweep_collision_rollouts:
        max_k = min(max_k, base_collision_per_step.shape[1])
    else:
        max_k = min(max_k, base_collision_per_step.shape[2])
    k_values = np.arange(max_k + 1)
    realism_scores = []
    collision_likelihoods = []
    offroad_likelihoods = []

    for k in k_values:
        override = base_collision_per_step.copy()
        if k:
            for a in range(override.shape[0]):
                if sweep_collision_rollouts:
                    for r in range(k):
                        idx = np.flatnonzero(valid_mask[a, r])
                        if idx.size:
                            override[a, r, idx[0]] = True
                else:
                    for r in range(override.shape[1]):
                        idx = np.flatnonzero(valid_mask[a, r])
                        override[a, r, idx[:k]] = True

        results = evaluator.compute_metrics(
            gt_trajectories,
            fake_simulated_trajectories,
            agent_state,
            road_edge_polylines,
            override_sim_collision_per_step=override,
        )
        realism_scores.append(results["realism_meta_score"].mean())
        collision_likelihoods.append(results["likelihood_collision_indication"].mean())
        offroad_likelihoods.append(results["likelihood_offroad_indication"].mean())

    return k_values, np.array(realism_scores), np.array(collision_likelihoods), np.array(offroad_likelihoods)


def plot_collision_sweep(k_values, scores, gt_baseline=None, random_no_collision=None, x_label=None):
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(k_values, scores, marker="o", color="tab:blue")
    if gt_baseline is not None:
        ax.axhline(gt_baseline, color="tab:green", linestyle="--", label="GT baseline")
    if random_no_collision is not None:
        ax.axhline(random_no_collision, color="tab:orange", linestyle="--", label="Random no collision")
    if gt_baseline is not None or random_no_collision is not None:
        ax.legend()
    ax.set_title("Realism vs collisions per agent")
    ax.set_xlabel(x_label or "# collisions per agent per rollout")
    ax.set_ylabel("Realism meta score")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig("wosac_collision_sweep.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return fig


def plot_likelihood_sweep(k_values, values, title, filename, gt_baseline=None, x_label=None):
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(k_values, values, marker="o", color="tab:blue")
    if gt_baseline is not None:
        ax.axhline(gt_baseline, color="tab:green", linestyle="--", label="GT baseline")
    if gt_baseline is not None:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label or "# collisions per agent per rollout")
    ax.set_ylabel("Likelihood")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return fig


def variance_sweep_from_ground_truth(config, vecenv, evaluator, plot_agent_idx=None):
    sigmas_xy = [0.0, 0.1, 0.2, 0.5, 1.0]
    sigmas_heading = [0.0, 0.02, 0.05, 0.1, 0.2]

    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)
    num_rollouts = config["eval"]["wosac_num_rollouts"]
    fake_simulated_trajectories = gt_trajectories.copy()
    for key in ["x", "y", "heading", "id"]:
        fake_simulated_trajectories[key] = np.repeat(gt_trajectories[key], num_rollouts, axis=1)
    fake_simulated_trajectories["id"] = fake_simulated_trajectories["id"][..., np.newaxis]
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()

    eval_mask = gt_trajectories["id"][:, 0] >= 0
    eval_sim_x = fake_simulated_trajectories["x"][eval_mask]
    eval_sim_y = fake_simulated_trajectories["y"][eval_mask]
    eval_sim_heading = fake_simulated_trajectories["heading"][eval_mask]
    eval_scenario_ids = gt_trajectories["scenario_id"][eval_mask]
    eval_agent_length = agent_state["length"][eval_mask]
    eval_agent_width = agent_state["width"][eval_mask]

    _, base_collision_per_step, _ = metrics.compute_interaction_features(
        fake_simulated_trajectories["x"],
        fake_simulated_trajectories["y"],
        fake_simulated_trajectories["heading"],
        gt_trajectories["scenario_id"],
        agent_state["length"],
        agent_state["width"],
        eval_mask,
        device=config["train"]["device"],
    )

    _, base_offroad_per_step = metrics.compute_map_features(
        eval_sim_x,
        eval_sim_y,
        eval_sim_heading,
        eval_scenario_ids,
        eval_agent_length,
        eval_agent_width,
        road_edge_polylines,
        device=config["train"]["device"],
    )

    realism_scores = []
    for sx, sh in zip(sigmas_xy, sigmas_heading):
        simulated = {k: v.copy() for k, v in fake_simulated_trajectories.items()}
        simulated["x"] += np.random.normal(0.0, sx, size=simulated["x"].shape)
        simulated["y"] += np.random.normal(0.0, sx, size=simulated["y"].shape)
        simulated["heading"] += np.random.normal(0.0, sh, size=simulated["heading"].shape)
        simulated["heading"] = (simulated["heading"] + np.pi) % (2 * np.pi) - np.pi

        if plot_agent_idx is not None:
            evaluator._quick_sanity_check(gt_trajectories, simulated, agent_idx=plot_agent_idx)
            os.replace(
                f"trajectory_comparison_agent_{plot_agent_idx}.png",
                f"trajectory_comparison_agent_{plot_agent_idx}_sigma_xy_{sx:.2f}_sigma_h_{sh:.2f}.png",
            )

        results = evaluator.compute_metrics(
            gt_trajectories,
            simulated,
            agent_state,
            road_edge_polylines,
            override_sim_collision_per_step=base_collision_per_step,
            override_sim_offroad_per_step=base_offroad_per_step,
        )
        realism_scores.append(results["realism_meta_score"].mean())

    return np.array(sigmas_xy), np.array(realism_scores)


def plot_variance_sweep(sigmas_xy, scores):
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(sigmas_xy, scores, marker="o", color="tab:blue")
    ax.set_title("Realism vs trajectory noise")
    ax.set_xlabel("sigma_xy (m) / sigma_heading (rad)")
    ax.set_ylabel("Realism meta score")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig("wosac_variance_sweep.png", dpi=300, bbox_inches="tight", facecolor="white")
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


def evaluate_random_policy(config, vecenv, evaluator, return_trajectories=False):
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

    if return_trajectories:
        return results, gt_trajectories, simulated_trajectories, agent_state, road_edge_polylines
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


def pipeline(
    env_name="puffer_drive",
    sweep_collision_rollouts=False,
    variance_sweep=False,
    variance_plot_agent=None,
):
    """Obtain WOSAC scores for various baselines and policies."""

    config = load_config(env_name)

    config["env"]["num_maps"] = 60
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
    config["env"]["save_data_to_disk"] = False

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
    # df_results_self_play = evaluate_rl_policy(
    #     config, vecenv, evaluator, "pufferlib/resources/drive/pufferdrive_weights.pt"
    # )  # POLICY_DIR + "/puffer_drive_sp_qld2z6tn.pt")
    # df_results_self_play["policy"] = "self_play_rl_base"

    # TODO: Guided self-play policy (guidance in rewards)
    # ...

    # TODO: Guided self-play policy (regularization)
    # ...

    # Baseline: Random policy
    (
        df_results_random,
        gt_traj_random,
        sim_traj_random,
        agent_state_random,
        road_edge_polylines_random,
    ) = evaluate_random_policy(config, vecenv, evaluator, return_trajectories=True)
    df_results_random["policy"] = "random"

    eval_mask = gt_traj_random["id"][:, 0] >= 0
    no_collision_override = np.zeros((int(np.sum(eval_mask)),) + sim_traj_random["x"].shape[1:], dtype=bool)
    df_results_random_no_collision = evaluator.compute_metrics(
        gt_traj_random,
        sim_traj_random,
        agent_state_random,
        road_edge_polylines_random,
        override_sim_collision_per_step=no_collision_override,
    )
    df_results_random_no_collision["policy"] = "random_no_collision"

    # Combine
    df = pd.concat(
        [
            df_results_gt,
            df_results_inferred_human,
            df_results_random,
            df_results_random_no_collision,
            # df_results_bc,
            # df_results_self_play,
        ],
        ignore_index=True,
    )

    df = df[COLUMN_ORDER]

    # Visualize
    plot_wosac_results(df)
    plot_realism_score_distributions(df)
    max_k = 10
    x_label = "colliding rollouts per agent" if sweep_collision_rollouts else "# collisions per agent per rollout"
    k_values, scores, collision_likelihoods, offroad_likelihoods = collision_sweep_from_ground_truth(
        config,
        vecenv,
        evaluator,
        max_k=max_k,
        sweep_collision_rollouts=sweep_collision_rollouts,
    )
    random_no_collision_score = df_results_random_no_collision["realism_meta_score"].mean()
    plot_collision_sweep(
        k_values,
        scores,
        gt_baseline=df_results_gt["realism_meta_score"].mean(),
        random_no_collision=random_no_collision_score,
        x_label=x_label,
    )
    plot_likelihood_sweep(
        k_values,
        collision_likelihoods,
        "Collision likelihood vs collisions per agent",
        "wosac_collision_likelihood_sweep.png",
        gt_baseline=df_results_gt["likelihood_collision_indication"].mean(),
        x_label=x_label,
    )
    plot_likelihood_sweep(
        k_values,
        offroad_likelihoods,
        "Offroad likelihood vs collisions per agent",
        "wosac_offroad_likelihood_sweep.png",
        gt_baseline=df_results_gt["likelihood_offroad_indication"].mean(),
        x_label=x_label,
    )
    if variance_sweep:
        sigmas_xy, variance_scores = variance_sweep_from_ground_truth(
            config, vecenv, evaluator, plot_agent_idx=variance_plot_agent
        )
        plot_variance_sweep(sigmas_xy, variance_scores)

    print(f"total agents: {df_results_gt['num_agents_per_scene'].sum().item()}")

    print(df.groupby("policy")["realism_meta_score"].mean())
    print("---")
    print(df.groupby("policy")["kinematic_metrics"].mean())
    print("---")
    print(df.groupby("policy")["interactive_metrics"].mean())
    print("---")
    print(df.groupby("policy")["map_based_metrics"].mean())

    breakpoint()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--collision-rollouts-sweep", action="store_true")
    parser.add_argument("--variance-sweep", action="store_true")
    parser.add_argument("--variance-plot-agent", type=int, default=None)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    pipeline(
        sweep_collision_rollouts=args.collision_rollouts_sweep,
        variance_sweep=args.variance_sweep,
        variance_plot_agent=args.variance_plot_agent,
    )
