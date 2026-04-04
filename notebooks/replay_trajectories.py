# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PufferDrive Trajectory Replay
#
# Load saved trajectory data from training checkpoints and visualize agent behavior.

# %% Configuration
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

# Point this to your experiment directory
RUN_DIR = "/scratch/ev2237/experiments/YOUR_RUN_HERE/puffer_drive_RUNID"
EPOCH = 100  # which checkpoint to load

# %%  Load trajectory data
traj_path = Path(RUN_DIR) / f"trajectories_{EPOCH:06d}.npz"
data = np.load(traj_path, allow_pickle=True)

print("Available keys:", list(data.keys()))
for k in data.keys():
    v = data[k]
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")

# %% Extract buffers
observations = data["observations"]  # (segments, horizon, obs_dim)
actions = data["actions"]  # (segments, horizon, ...)
rewards = data["rewards"]  # (segments, horizon)
terminals = data["terminals"]  # (segments, horizon)
truncations = data["truncations"]  # (segments, horizon)
is_invalid = data["is_invalid_step"]  # (segments, horizon)
values = data["values"]  # (segments, horizon)

segments, horizon = rewards.shape[:2]
print(f"Segments: {segments}, Horizon: {horizon}")
print(f"Total timesteps: {segments * horizon}")

# %% [markdown]
# ## Episode Reconstruction
#
# Split the flat trajectory data into individual episodes using terminal/truncation signals.

# %% Reconstruct episodes
flat_rewards = rewards.reshape(-1)
flat_terminals = terminals.reshape(-1)
flat_truncations = truncations.reshape(-1)
flat_invalid = is_invalid.reshape(-1)
flat_values = values.reshape(-1)
flat_obs = observations.reshape(-1, observations.shape[-1])

done_mask = (flat_terminals + flat_truncations).clip(max=1)
valid_mask = flat_invalid == 0

# Find episode boundaries
episode_ends = np.where(done_mask > 0)[0]
episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])

episode_returns = []
episode_lengths = []
for start, end in zip(episode_starts, episode_ends):
    ep_rewards = flat_rewards[start : end + 1]
    ep_valid = valid_mask[start : end + 1]
    ep_return = ep_rewards[ep_valid].sum()
    ep_length = ep_valid.sum()
    episode_returns.append(ep_return)
    episode_lengths.append(ep_length)

episode_returns = np.array(episode_returns)
episode_lengths = np.array(episode_lengths)
print(f"Found {len(episode_returns)} episodes")
print(f"Mean return: {episode_returns.mean():.2f}, Mean length: {episode_lengths.mean():.1f}")

# %% [markdown]
# ## Aggregate Statistics

# %% Episode return distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(episode_returns, bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Episode Return")
axes[0].set_ylabel("Count")
axes[0].set_title("Episode Return Distribution")
axes[0].axvline(episode_returns.mean(), color="red", linestyle="--", label=f"Mean: {episode_returns.mean():.2f}")
axes[0].legend()

axes[1].hist(episode_lengths, bins=50, edgecolor="black", alpha=0.7, color="orange")
axes[1].set_xlabel("Episode Length (valid steps)")
axes[1].set_ylabel("Count")
axes[1].set_title("Episode Length Distribution")
axes[1].axvline(episode_lengths.mean(), color="red", linestyle="--", label=f"Mean: {episode_lengths.mean():.1f}")
axes[1].legend()

# Reward per step over time
window = max(1, len(flat_rewards) // 200)
smoothed_rewards = np.convolve(flat_rewards * valid_mask, np.ones(window) / window, mode="valid")
axes[2].plot(smoothed_rewards, alpha=0.7)
axes[2].set_xlabel("Timestep")
axes[2].set_ylabel("Reward (smoothed)")
axes[2].set_title("Reward Over Time in Buffer")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Per-Agent Timeline
#
# Look at individual agent trajectories within segments.

# %% Plot a single segment's timeline
seg_idx = 0  # which segment to inspect

seg_rewards = rewards[seg_idx]
seg_invalid = is_invalid[seg_idx]
seg_values = values[seg_idx]
seg_terminals = terminals[seg_idx]
seg_obs = observations[seg_idx]

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Rewards
axes[0].plot(seg_rewards, label="reward", alpha=0.8)
axes[0].fill_between(range(horizon), 0, seg_rewards, alpha=0.3)
axes[0].set_ylabel("Reward")
axes[0].legend()

# Values
axes[1].plot(seg_values, label="V(s)", color="green", alpha=0.8)
axes[1].set_ylabel("Value")
axes[1].legend()

# Speed (obs[3] = signed_speed / MAX_SPEED)
speed = seg_obs[:, 3] * 100  # denormalize
axes[2].plot(speed, label="speed (m/s)", color="purple", alpha=0.8)
axes[2].set_ylabel("Speed (m/s)")
axes[2].legend()

# Invalid steps + terminals
axes[3].fill_between(range(horizon), 0, seg_invalid, alpha=0.3, color="red", label="invalid")
axes[3].fill_between(range(horizon), 0, seg_terminals, alpha=0.3, color="blue", label="terminal")
axes[3].set_ylabel("Flags")
axes[3].set_xlabel("Step within segment")
axes[3].legend()

plt.suptitle(f"Segment {seg_idx} Timeline")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Goal Direction Visualization
#
# Plot the goal direction from ego observations over time.

# %% Goal direction over a segment
seg_idx = 0
seg_obs = observations[seg_idx]
seg_invalid = is_invalid[seg_idx].astype(bool)

# obs[0:2] = rel_goal_x/y * 0.005
goal_x = seg_obs[:, 0] / 0.005  # denormalize to meters
goal_y = seg_obs[:, 1] / 0.005

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
valid_steps = ~seg_invalid
colors = np.arange(horizon)[valid_steps]
sc = ax.scatter(goal_x[valid_steps], goal_y[valid_steps], c=colors, cmap="viridis", s=10, alpha=0.7)
ax.set_xlabel("Goal X (ego frame, meters)")
ax.set_ylabel("Goal Y (ego frame, meters)")
ax.set_title(f"Goal Direction Over Time (Segment {seg_idx})")
ax.set_aspect("equal")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
plt.colorbar(sc, label="Step")
plt.show()

# %% [markdown]
# ## Action Distribution

# %% Action frequency heatmap
flat_actions = actions.reshape(-1)
flat_valid = valid_mask

valid_actions = flat_actions[flat_valid]
n_actions = int(valid_actions.max()) + 1

# Split into time bins
n_bins = 20
bin_size = len(valid_actions) // n_bins
action_hist = np.zeros((n_bins, n_actions))
for b in range(n_bins):
    start = b * bin_size
    end = start + bin_size
    for a in range(n_actions):
        action_hist[b, a] = (valid_actions[start:end] == a).sum()
    action_hist[b] /= action_hist[b].sum() + 1e-8

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
im = ax.imshow(action_hist.T, aspect="auto", cmap="hot", interpolation="nearest")
ax.set_xlabel("Time bin")
ax.set_ylabel("Action index")
ax.set_title("Action Distribution Over Time")
plt.colorbar(im, label="Frequency")
plt.show()

# %% [markdown]
# ## Lane Alignment Over Time

# %% Lane metrics from observations
seg_idx = 0
seg_obs = observations[seg_idx]
seg_invalid = is_invalid[seg_idx].astype(bool)

# obs[14] = lane_center_dist, obs[15] = cos(lane_angle), obs[16] = sin(lane_angle)
lane_center = seg_obs[:, 14]
lane_cos = seg_obs[:, 15]
lane_sin = seg_obs[:, 16]

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

axes[0].plot(lane_center, alpha=0.8)
axes[0].fill_between(range(horizon), 0, seg_invalid * lane_center.max(), alpha=0.2, color="red")
axes[0].set_ylabel("Lane Center Dist")
axes[0].set_title("Lane Alignment Metrics")

axes[1].plot(lane_cos, label="cos(theta)", alpha=0.8)
axes[1].axhline(1.0, color="green", linestyle="--", alpha=0.3, label="perfect alignment")
axes[1].set_ylabel("cos(lane angle)")
axes[1].legend()

axes[2].plot(lane_sin, label="sin(theta)", alpha=0.8, color="orange")
axes[2].axhline(0, color="gray", linestyle="--", alpha=0.3)
axes[2].set_ylabel("sin(lane angle)")
axes[2].set_xlabel("Step")
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compare Two Checkpoints

# %% Load and compare
EPOCH_A = 100
EPOCH_B = 500

try:
    data_a = np.load(Path(RUN_DIR) / f"trajectories_{EPOCH_A:06d}.npz")
    data_b = np.load(Path(RUN_DIR) / f"trajectories_{EPOCH_B:06d}.npz")

    def compute_stats(d):
        r = d["rewards"].reshape(-1)
        v = d["is_invalid_step"].reshape(-1) == 0
        t = (d["terminals"].reshape(-1) + d["truncations"].reshape(-1)).clip(max=1)
        ends = np.where(t > 0)[0]
        starts = np.concatenate([[0], ends[:-1] + 1])
        returns = [r[s : e + 1][v[s : e + 1] == 1].sum() for s, e in zip(starts, ends)]
        speed = d["observations"].reshape(-1, d["observations"].shape[-1])[:, 3] * 100
        return {"returns": np.array(returns), "mean_speed": speed[v].mean()}

    stats_a = compute_stats(data_a)
    stats_b = compute_stats(data_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(stats_a["returns"], bins=30, alpha=0.5, label=f"Epoch {EPOCH_A}")
    axes[0].hist(stats_b["returns"], bins=30, alpha=0.5, label=f"Epoch {EPOCH_B}")
    axes[0].set_xlabel("Episode Return")
    axes[0].set_title("Return Distribution Comparison")
    axes[0].legend()

    axes[1].bar(
        [f"Epoch {EPOCH_A}", f"Epoch {EPOCH_B}"],
        [stats_a["mean_speed"], stats_b["mean_speed"]],
        color=["tab:blue", "tab:orange"],
    )
    axes[1].set_ylabel("Mean Speed (m/s)")
    axes[1].set_title("Average Speed Comparison")

    plt.tight_layout()
    plt.show()
except FileNotFoundError as e:
    print(f"Could not load comparison data: {e}")
