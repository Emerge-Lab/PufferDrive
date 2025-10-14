"""WOSAC evaluation class for PufferDrive."""

import torch
import time
import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt

import pufferlib
from pufferlib.ocean.drive.drive import Drive


class WOSACEvaluator:
    """Evaluates policys on the Waymo Open Sim Agent Challenge (WOSAC) in PufferDrive.

    Leaderboard: https://waymo.com/open/challenges/2025/sim-agents/
    """

    def __init__(self, config: Dict):
        self.config = config
        self.num_steps = 91
        self.show_dashboard = config.get("wosac", {}).get("dashboard", False)
        self.num_rollouts = config.get("wosac", {}).get("num_rollouts", 32)

    def collect_simulated_trajectories(self, args, vecenv, policy):
        """Roll out policy in vecenv and collect trajectories.
        Returns:
            trajectories: dict with keys 'x', 'y', 'z', 'heading' each of shape
                (num_agents, num_rollouts, num_steps)
        """

        driver = vecenv.driver_env
        num_agents = vecenv.observation_space.shape[0]
        device = args["train"]["device"]

        trajectories = {
            "x": np.zeros((num_agents, self.num_rollouts, self.num_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.num_rollouts, self.num_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.num_rollouts, self.num_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.num_rollouts, self.num_steps), dtype=np.float32),
            "id": np.zeros((num_agents, self.num_rollouts, self.num_steps), dtype=np.int32),
        }

        for rollout_idx in range(self.num_rollouts):
            obs, info = vecenv.reset()
            state = {}
            if args["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                )

            for time_idx in range(self.num_steps):
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
                    action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)

                if isinstance(logits, torch.distributions.Normal):
                    action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)

                obs, _, _, _, _ = vecenv.step(action_np)

        if self.show_dashboard:
            self._display_dashboard(trajectories)

        import pdb

        pdb.set_trace()

        return trajectories

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

    def collect_ground_truth_data(self):
        """Collect ground truth data for evaluation."""
        pass

    def compute_metrics(self, x_hat, y_hat, z_hat, heading_hat):
        """Compute evaluation metrics given predicted and ground truth data.

        Args:
            x_hat, y_hat, z_hat, heading_hat: Predicted data from the policy rollouts.
            x, y, z, heading: Ground truth data.
        """
        pass
