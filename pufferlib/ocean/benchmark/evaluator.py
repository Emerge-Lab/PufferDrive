"""WOSAC evaluation class for PufferDrive."""

import torch
import time
import numpy as np
import pandas as pd
from pprint import pprint
from typing import Dict, Optional
import matplotlib.pyplot as plt

import pufferlib
from pufferlib.ocean.benchmark import metrics
from pufferlib.ocean.benchmark import estimators


class WOSACEvaluator:
    """Evaluates policys on the Waymo Open Sim Agent Challenge (WOSAC) in PufferDrive.

    Leaderboard: https://waymo.com/open/challenges/2025/sim-agents/
    """

    def __init__(self, config: Dict):
        self.config = config
        self.num_steps = 91  # Hardcoded for WOSAC (9.1s at 10Hz)
        self.init_steps = 10  # Initial steps to skip (1s)
        self.sim_steps = self.num_steps - self.init_steps
        self.show_dashboard = config.get("wosac", {}).get("dashboard", False)
        self.num_rollouts = config.get("wosac", {}).get("num_rollouts", 32)

    def collect_ground_truth_trajectories(self, puffer_env):
        """Collect ground truth data for evaluation.
        Returns:
            trajectories: dict with keys 'x', 'y', 'z', 'heading', 'id'
                        each of shape (num_agents, 1, num_steps) for trajectory data
        """
        return puffer_env.get_ground_truth_trajectories()

    def collect_simulated_trajectories(self, args, puffer_env, policy):
        """Roll out policy in env and collect trajectories.
        Returns:
            trajectories: dict with keys 'x', 'y', 'z', 'heading' each of shape
                (num_agents, num_rollouts, num_steps)
        """

        driver = puffer_env.driver_env
        num_agents = puffer_env.observation_space.shape[0]
        device = args["train"]["device"]

        trajectories = {
            "x": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "id": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.int32),
        }

        for rollout_idx in range(self.num_rollouts):
            print(f"\rCollecting rollout {rollout_idx + 1}/{self.num_rollouts}...", end="", flush=True)
            obs, info = puffer_env.reset()
            state = {}
            if args["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                )

            for time_idx in range(self.sim_steps):
                # Get global state
                agent_state = driver.get_global_agent_state()
                trajectories["x"][:, rollout_idx, time_idx] = agent_state["x"]
                trajectories["y"][:, rollout_idx, time_idx] = agent_state["y"]
                trajectories["z"][:, rollout_idx, time_idx] = agent_state["z"]
                trajectories["heading"][:, rollout_idx, time_idx] = agent_state["heading"]
                trajectories["id"][:, rollout_idx, time_idx] = agent_state["id"]

                # Step policy
                with torch.no_grad():
                    ob_tensor = torch.as_tensor(obs).to(device)
                    logits, value = policy.forward_eval(ob_tensor, state)
                    action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                    action_np = action.cpu().numpy().reshape(puffer_env.action_space.shape)

                if isinstance(logits, torch.distributions.Normal):
                    action_np = np.clip(action_np, puffer_env.action_space.low, puffer_env.action_space.high)

                obs, _, _, _, _ = puffer_env.step(action_np)

        if self.show_dashboard:
            self._display_dashboard(trajectories)

        return trajectories

    def compute_metrics(
        self,
        ground_truth_trajectories: Dict,
        simulated_trajectories: Dict,
    ) -> Dict:
        """Compute realism metrics comparing simulated and ground truth trajectories.

        Args:
            ground_truth_trajectories: Dict with keys ['x', 'y', 'z', 'heading', 'id']
                                Each trajectory has shape (n_agents, n_rollouts, n_steps)
            simulated_trajectories: Dict with same keys plus 'scenario_id'
                                shape (n_agents, n_steps) for trajectories
                                shape (n_agents,) for id
                                list of length n_agents for scenario_id

        Note: z-position not used in current metrics.

        Returns:
            Dictionary with scores per scenario_id
        """
        # Ensure the id order matches exactly for simulated and ground truth
        assert np.array_equal(simulated_trajectories["id"][:, 0:1, 0], ground_truth_trajectories["id"]), (
            "Agent IDs don't match between simulated and ground truth trajectories"
        )

        # Extract trajectories
        sim_x = simulated_trajectories["x"]
        sim_y = simulated_trajectories["y"]
        sim_heading = simulated_trajectories["heading"]
        ref_x = ground_truth_trajectories["x"]
        ref_y = ground_truth_trajectories["y"]
        ref_heading = ground_truth_trajectories["heading"]
        ref_valid = ground_truth_trajectories["valid"]

        # Compute features
        # Kinematics-related features
        sim_linear_speed, sim_linear_accel, sim_angular_speed, sim_angular_accel = metrics.compute_kinematic_features(
            sim_x, sim_y, sim_heading
        )

        ref_linear_speed, ref_linear_accel, ref_angular_speed, ref_angular_accel = metrics.compute_kinematic_features(
            ref_x, ref_y, ref_heading
        )

        # Get the log speed (linear and angular) validity. Since this is computed by
        # a delta between steps i-1 and i+1, we verify that both of these are
        # valid (logical and).
        speed_validity, acceleration_validity = metrics.compute_kinematic_validity(ref_valid)

        # Compute realism metrics
        # Average Displacement Error (ADE) and minADE
        # Note: This metric is not included in the scoring meta-metric, as per WOSAC rules.
        ade, min_ade = metrics.compute_displacement_error(sim_x, sim_y, ref_x, ref_y, ref_valid)

        # Log-likelihood metrics
        # Kinematic features log-likelihoods
        linear_speed_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_linear_speed,
            sim_values=sim_linear_speed,
            treat_timesteps_independently=True,
            min_val=0.0,
            max_val=25.0,
            num_bins=10,
            sanity_check=True,
        )

        # Compute final normalized scores
        # Each of these metrics is in [0, 1], higher is better

        # Input: [num_agents, num_steps]
        # TODO: Check inputs/outputs from ll func above
        speed_likelihood = np.exp(
            metrics._reduce_average_with_validity(linear_speed_log_likelihood, speed_validity[:, 0, :])
        )

        breakpoint()

        # Get agent IDs and scenario IDs
        agent_ids = ground_truth_trajectories["id"]
        scenario_ids = ground_truth_trajectories["scenario_id"]

        df = pd.DataFrame(
            {
                "agent_id": agent_ids.flatten(),
                "scenario_id": scenario_ids.flatten(),
                "ade": ade,
                "min_ade": min_ade,
                "likelihood_speed": speed_likelihood,
            }
        )

        # Aggregate results per scenario_id
        results = df.groupby("scenario_id")[["ade", "min_ade"]].mean()

        print(results)

        return results

    def _quick_sanity_check(self, gt_trajectories, simulated_trajectories):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        agent_idx = 0  # Visualize the first agent
        axs[0].set_title(f"Agent ID: {simulated_trajectories['id'][agent_idx, 0][0]}")
        axs[0].scatter(
            simulated_trajectories["x"][agent_idx, :, :],
            simulated_trajectories["y"][agent_idx, :, :],
            color="b",
            alpha=0.1,
            label="Simulated",
        )
        axs[0].scatter(
            gt_trajectories["x"][agent_idx, :, :],
            gt_trajectories["y"][agent_idx, :, :],
            color="g",
            label="Ground Truth",
        )
        axs[0].scatter(
            gt_trajectories["x"][agent_idx, 0, 0],
            gt_trajectories["y"][agent_idx, 0, 0],
            color="purple",
            marker="*",
            s=200,
            label="GT start",
            zorder=5,
            alpha=0.5,
        )
        axs[0].scatter(
            simulated_trajectories["x"][agent_idx, :, 0],
            simulated_trajectories["y"][agent_idx, :, 0],
            color="purple",
            marker="*",
            s=200,
            label="Agent start",
            zorder=5,
            alpha=0.5,
        )
        axs[0].set_xlabel("X Position")
        axs[0].set_ylabel("Y Position")
        axs[0].legend()

        axs[1].set_title(f"Heading timeseries; ID: {simulated_trajectories['id'][agent_idx, 0][0]}")
        time_steps = list(range(self.sim_steps))
        for r in range(self.num_rollouts):
            axs[1].plot(
                time_steps, simulated_trajectories["heading"][agent_idx, r, :], color="b", alpha=0.1, label="Simulated"
            )
        axs[1].plot(time_steps, gt_trajectories["heading"][agent_idx, :, :].T, color="g", label="Ground Truth")
        axs[1].set_xlabel("Time Step")
        plt.savefig("trajectory_comparison.png")

    def _display_dashboard(self, trajectories):
        """Display sanity checks and visualizations for collected trajectories."""
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)

        print("\n" + "=" * 70)
        print("🚗 WOSAC DASHBOARD")
        print("=" * 70)

        # Flatten trajectories over rollouts and steps for analysis
        # Original shape: (num_agents, num_rollouts, num_steps)
        # Flattened shape for stats: (num_agents, num_rollouts * num_steps)
        num_agents, num_rollouts, num_steps = trajectories["x"].shape

        # Reshape to (num_agents, num_rollouts * num_steps)
        x_flat = trajectories["x"].reshape(num_agents, -1)
        y_flat = trajectories["y"].reshape(num_agents, -1)
        z_flat = trajectories["z"].reshape(num_agents, -1)
        heading_flat = trajectories["heading"].reshape(num_agents, -1)

        # ==================== Summary Statistics ====================
        print(f"\n📊 Data Summary:")
        print(f"  • Original shape: ({num_agents} agents, {num_rollouts} rollouts, {num_steps} steps)")
        print(f"  • Total samples: {num_agents * num_rollouts * num_steps}")
        print(f"  • X range: [{x_flat.min():.2f}, {x_flat.max():.2f}] m")
        print(f"  • Y range: [{y_flat.min():.2f}, {y_flat.max():.2f}] m")
        print(f"  • Z range: [{z_flat.min():.2f}, {z_flat.max():.2f}] m")
        print(f"  • Heading range: [{heading_flat.min():.2f}, {heading_flat.max():.2f}] rad")

        # ==================== Sanity Checks ====================
        print(f"\n🔍 Sanity Checks:")

        # Check for static agents (across all rollouts)
        x_std = np.std(x_flat, axis=1)
        y_std = np.std(y_flat, axis=1)
        static_threshold = 0.01
        num_static = np.sum((x_std < static_threshold) & (y_std < static_threshold))
        print(f"  • Static agents (std < {static_threshold}): {num_static}/{num_agents}")

        # Check for teleportation (large jumps) - compute for each rollout separately
        teleport_threshold = 50.0
        total_teleports = 0
        max_jump_overall = 0

        for rollout_idx in range(num_rollouts):
            dx = np.diff(trajectories["x"][:, rollout_idx, :], axis=1)
            dy = np.diff(trajectories["y"][:, rollout_idx, :], axis=1)
            jumps = np.sqrt(dx**2 + dy**2)
            total_teleports += np.sum(jumps > teleport_threshold)
            max_jump_overall = max(max_jump_overall, np.max(jumps))

        print(f"  • Potential teleports (Δ > {teleport_threshold}m): {total_teleports}")
        print(f"  • Max single-step displacement: {max_jump_overall:.2f} m")

        # Check for invalid positions
        num_invalid_x = np.sum(np.isnan(x_flat) | np.isinf(x_flat))
        num_invalid_y = np.sum(np.isnan(y_flat) | np.isinf(y_flat))
        print(f"  • Invalid X positions (NaN/Inf): {num_invalid_x}")
        print(f"  • Invalid Y positions (NaN/Inf): {num_invalid_y}")

        # Compute speed statistics (average across rollouts)
        dt = 0.1  # Time step in seconds
        all_speeds = []
        for rollout_idx in range(num_rollouts):
            dx = np.diff(trajectories["x"][:, rollout_idx, :], axis=1)
            dy = np.diff(trajectories["y"][:, rollout_idx, :], axis=1)
            speeds = np.sqrt(dx**2 + dy**2) / dt
            all_speeds.append(speeds)

        all_speeds = np.concatenate(all_speeds, axis=1)

        print(f"\n📈 Speed Statistics:")
        print(f"  • Mean: {np.mean(all_speeds):.2f} m/s")
        print(f"  • Median: {np.median(all_speeds):.2f} m/s")
        print(f"  • Std: {np.std(all_speeds):.2f} m/s")
        print(f"  • Max: {np.max(all_speeds):.2f} m/s")
        print(f"  • 95th percentile: {np.percentile(all_speeds, 95):.2f} m/s")

        # Heading change statistics
        all_dheading = []
        for rollout_idx in range(num_rollouts):
            dheading = np.diff(trajectories["heading"][:, rollout_idx, :], axis=1)
            dheading = np.arctan2(np.sin(dheading), np.cos(dheading))
            all_dheading.append(dheading)

        all_dheading = np.concatenate(all_dheading, axis=1)

        print(f"\n🧭 Heading Change Statistics:")
        print(f"  • Mean abs change: {np.mean(np.abs(all_dheading)):.3f} rad/step")
        print(f"  • Max abs change: {np.max(np.abs(all_dheading)):.3f} rad/step")
        print(f"  • Std: {np.std(all_dheading):.3f} rad/step")

        # ==================== Visualizations ====================
        colors = sns.color_palette("husl", n_colors=min(num_agents, 20))
        if num_agents > 20:
            colors = sns.color_palette("Spectral", n_colors=num_agents)

        fig = plt.figure(figsize=(18, 11))

        # Plot 1: X distribution
        ax1 = plt.subplot(3, 3, 1)
        sns.histplot(
            x_flat.flatten(),
            bins=50,
            kde=True,
            color=sns.color_palette("deep")[0],
            edgecolor="white",
            linewidth=0.5,
            ax=ax1,
        )
        ax1.set_title(r"$x$ position distribution", fontsize=13, fontweight="bold", pad=10)
        ax1.set_xlabel(r"$x$ (m)", fontsize=11)
        ax1.set_ylabel("Frequency", fontsize=11)
        ax1.legend(fontsize=10, frameon=True, fancybox=True)

        # Plot 2: Y distribution
        ax2 = plt.subplot(3, 3, 2)
        sns.histplot(
            y_flat.flatten(),
            bins=50,
            kde=True,
            color=sns.color_palette("deep")[2],
            edgecolor="white",
            linewidth=0.5,
            ax=ax2,
        )
        ax2.set_title(r"$y$ position distribution", fontsize=13, fontweight="bold", pad=10)
        ax2.set_xlabel(r"$y$ (m)", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.legend(fontsize=10, frameon=True, fancybox=True)

        # Plot 3: Z distribution
        ax3 = plt.subplot(3, 3, 3)
        sns.histplot(
            z_flat.flatten(),
            bins=50,
            kde=True,
            color=sns.color_palette("deep")[3],
            edgecolor="white",
            linewidth=0.5,
            ax=ax3,
        )
        ax3.set_title(r"$z$ position distribution", fontsize=13, fontweight="bold", pad=10)
        ax3.set_xlabel(r"$z$ (m)", fontsize=11)
        ax3.set_ylabel("Frequency", fontsize=11)
        ax3.legend(fontsize=10, frameon=True, fancybox=True)

        # Plot 4: Heading distribution
        ax4 = plt.subplot(3, 3, 4)
        sns.histplot(
            heading_flat.flatten(),
            bins=50,
            kde=True,
            color=sns.color_palette("deep")[4],
            edgecolor="white",
            linewidth=0.5,
            ax=ax4,
        )
        mean_heading = np.mean(heading_flat)
        ax4.axvline(
            mean_heading, color="#e74c3c", linestyle="--", linewidth=2.5, label=f"Mean: {mean_heading:.2f}", zorder=10
        )
        ax4.set_title("Heading Distribution", fontsize=13, fontweight="bold", pad=10)
        ax4.set_xlabel("Heading (rad)", fontsize=11)
        ax4.set_ylabel("Frequency", fontsize=11)
        ax4.set_xlim([-np.pi, np.pi])
        ax4.legend(fontsize=10, frameon=True, fancybox=True)

        # Plot 5: X position over time (averaged across rollouts)
        ax5 = plt.subplot(3, 3, 5)
        timesteps = np.arange(num_steps)

        # Average over rollouts for cleaner visualization
        x_mean_rollouts = np.mean(trajectories["x"], axis=1)  # (num_agents, num_steps)

        for i in range(min(num_agents, 10)):  # Show max 10 agents for clarity
            color = colors[i % len(colors)]
            ax5.plot(
                timesteps,
                x_mean_rollouts[i, :],
                alpha=0.7,
                linewidth=2,
                color=color,
                label=f"Agent {i}" if num_agents <= 5 else None,
            )

        ax5.set_title(f"X Position Over Time", fontsize=13, fontweight="bold", pad=10)
        ax5.set_xlabel("Timestep", fontsize=11)
        ax5.set_ylabel("X (m)", fontsize=11)
        if num_agents <= 5:
            ax5.legend(fontsize=9, frameon=True, fancybox=True)

        # Plot 6: Y position over time (averaged across rollouts)
        ax6 = plt.subplot(3, 3, 6)
        y_mean_rollouts = np.mean(trajectories["y"], axis=1)  # (num_agents, num_steps)

        for i in range(min(num_agents, 10)):
            color = colors[i % len(colors)]
            ax6.plot(
                timesteps,
                y_mean_rollouts[i, :],
                alpha=0.7,
                linewidth=2,
                color=color,
                label=f"Agent {i}" if num_agents <= 5 else None,
            )

        ax6.set_title(f"Y Position Over Time", fontsize=13, fontweight="bold", pad=10)
        ax6.set_xlabel("Timestep", fontsize=11)
        ax6.set_ylabel("Y (m)", fontsize=11)
        if num_agents <= 5:
            ax6.legend(fontsize=9, frameon=True, fancybox=True)

        # Plot 7: 2D trajectories (X vs Y) - show all rollouts
        ax7 = plt.subplot(3, 3, 7)

        for agent_idx in range(min(num_agents, 10)):
            color = colors[agent_idx % len(colors)]
            for rollout_idx in range(num_rollouts):
                alpha = 0.3 if num_rollouts > 1 else 0.7
                ax7.plot(
                    trajectories["x"][agent_idx, rollout_idx, :],
                    trajectories["y"][agent_idx, rollout_idx, :],
                    alpha=alpha,
                    linewidth=1.5,
                    color=color,
                )
                # Mark start position
                ax7.scatter(
                    trajectories["x"][agent_idx, rollout_idx, 0],
                    trajectories["y"][agent_idx, rollout_idx, 0],
                    color=color,
                    s=80,
                    marker="o",
                    edgecolors="white",
                    linewidths=2,
                    zorder=5,
                )
                # Mark end position
                ax7.scatter(
                    trajectories["x"][agent_idx, rollout_idx, -1],
                    trajectories["y"][agent_idx, rollout_idx, -1],
                    color=color,
                    s=80,
                    marker="s",
                    edgecolors="white",
                    linewidths=2,
                    zorder=5,
                )

        ax7.set_title(f"2D Trajectories", fontsize=13, fontweight="bold", pad=10)
        ax7.set_xlabel("X (m)", fontsize=11)
        ax7.set_ylabel("Y (m)", fontsize=11)
        ax7.set_aspect("equal", adjustable="box")

        # Plot 8: Speed distribution
        ax8 = plt.subplot(3, 3, 8)
        sns.histplot(
            all_speeds.flatten(),
            bins=50,
            kde=True,
            color=sns.color_palette("deep")[5],
            edgecolor="white",
            linewidth=0.5,
            ax=ax8,
        )
        median_speed = np.median(all_speeds)
        mean_speed = np.mean(all_speeds)
        ax8.axvline(
            median_speed,
            color="#e74c3c",
            linestyle="--",
            linewidth=2.5,
            label=f"Median: {median_speed:.2f} m/s",
            zorder=10,
        )
        ax8.axvline(
            mean_speed, color="#f39c12", linestyle="--", linewidth=2.5, label=f"Mean: {mean_speed:.2f} m/s", zorder=10
        )
        ax8.set_title("Speed Distribution", fontsize=13, fontweight="bold", pad=10)
        ax8.set_xlabel("Speed (m/s)", fontsize=11)
        ax8.set_ylabel("Frequency", fontsize=11)
        ax8.legend(fontsize=10, frameon=True, fancybox=True)

        # Plot 9: Position heatmap
        ax9 = plt.subplot(3, 3, 9)
        h = ax9.hist2d(x_flat.flatten(), y_flat.flatten(), bins=50, cmap="YlOrRd", cmin=1)
        cbar = plt.colorbar(h[3], ax=ax9)
        cbar.set_label("Density", fontsize=11)
        ax9.set_title("Position Density Heatmap", fontsize=13, fontweight="bold", pad=10)
        ax9.set_xlabel("X (m)", fontsize=11)
        ax9.set_ylabel("Y (m)", fontsize=11)
        ax9.set_aspect("equal", adjustable="box")

        plt.suptitle("WOSAC metrics dashboard", fontsize=18, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save to file
        output_path = "wosac_dashboard.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"\n💾 Dashboard saved to: {output_path}")

        # Display if in interactive mode
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            pass  # Non-interactive environment

        print("=" * 70 + "\n")
