# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trajectory Visualization
#
# Visualize saved simulation trajectories from training checkpoints.

# %% Configuration
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TRAJ_PATH = "/tmp/traj_latest.npz"

# %% Load data
data = np.load(TRAJ_PATH, allow_pickle=True)

traj_x = data["traj_x"]
traj_y = data["traj_y"]
traj_heading = data["traj_heading"]
traj_lengths = data["traj_lengths"]
map_ids = data["map_ids"]
map_files = data["map_files"]
rewards = data["rewards"]
terminals = data["terminals"]
truncations = data["truncations"]
actions = data["actions"]
is_invalid = data["is_invalid_step"]

print(f"Total agents: {len(traj_lengths)}")
print(f"Agents with data: {np.count_nonzero(traj_lengths)}")
valid_lengths = traj_lengths[traj_lengths > 0]
print(f"Mean traj length: {valid_lengths.mean():.1f}")
print(f"Max traj length: {traj_lengths.max()}")
print(f"Maps: {map_files}")
print(f"Map distribution: {np.bincount(map_ids)}")

# %% [markdown]
# ## Trajectory Length Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(valid_lengths, bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Trajectory Length (steps)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Trajectory Length Distribution (mean={valid_lengths.mean():.1f})")
axes[0].axvline(valid_lengths.mean(), color="red", linestyle="--")

for mid in range(len(map_files)):
    mask = (map_ids == mid) & (traj_lengths > 0)
    if mask.sum() > 0:
        axes[1].hist(traj_lengths[mask], bins=30, alpha=0.5, label=f"Map {mid}: {Path(str(map_files[mid])).stem}")
axes[1].set_xlabel("Trajectory Length")
axes[1].set_title("Length Distribution per Map")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Spatial Trajectories per Map

# %%
n_maps = len(map_files)
fig, axes = plt.subplots(1, n_maps, figsize=(7 * n_maps, 7))
if n_maps == 1:
    axes = [axes]

for mid in range(n_maps):
    ax = axes[mid]
    mask = (map_ids == mid) & (traj_lengths > 5)
    agent_indices = np.where(mask)[0]

    # Sort by length, show longest trajectories
    sorted_idx = agent_indices[np.argsort(traj_lengths[agent_indices])[::-1]]
    n_show = min(50, len(sorted_idx))

    for i in sorted_idx[:n_show]:
        length = traj_lengths[i]
        x = traj_x[i, :length]
        y = traj_y[i, :length]
        ax.plot(x, y, alpha=0.4, linewidth=0.8)
        ax.plot(x[0], y[0], "go", markersize=3, alpha=0.5)
        ax.plot(x[-1], y[-1], "rx", markersize=3, alpha=0.5)

    ax.set_aspect("equal")
    ax.set_title(f"Map {mid}: {Path(str(map_files[mid])).stem}\n({mask.sum()} agents, showing {n_show})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Detailed Single-Agent Trajectory

# %%
best_idx = np.argmax(traj_lengths)
length = traj_lengths[best_idx]
print(f"Agent {best_idx}: length={length}, map={map_ids[best_idx]}")

x = traj_x[best_idx, :length]
y = traj_y[best_idx, :length]
h = traj_heading[best_idx, :length]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# XY trajectory colored by time
ax = axes[0, 0]
sc = ax.scatter(x, y, c=np.arange(length), cmap="viridis", s=5, alpha=0.8)
ax.plot(x[0], y[0], "go", markersize=10, label="start")
ax.plot(x[-1], y[-1], "rx", markersize=10, label="end")
ax.set_aspect("equal")
ax.set_title(f"Agent {best_idx} Trajectory (colored by time)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend()
plt.colorbar(sc, ax=ax, label="Step")

# Speed over time
ax = axes[0, 1]
if length > 1:
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2) / 0.1  # dt=0.1
    ax.plot(speed, alpha=0.8)
    ax.set_ylabel("Speed (m/s)")
    ax.set_xlabel("Step")
    ax.set_title(f"Speed (mean={speed.mean():.1f} m/s)")

# Heading over time
ax = axes[1, 0]
ax.plot(np.degrees(h), alpha=0.8)
ax.set_xlabel("Step")
ax.set_ylabel("Heading (degrees)")
ax.set_title("Heading Over Time")

# Yaw rate
ax = axes[1, 1]
if length > 1:
    dh = np.diff(h)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    ax.plot(np.degrees(dh) / 0.1, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Yaw rate (deg/s)")
    ax.set_title("Yaw Rate")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Action Distribution

# %%
flat_actions = actions.reshape(-1)
flat_valid = is_invalid.reshape(-1) == 0
valid_actions = flat_actions[flat_valid]

jerk_long = [-15, -4, 0, 4]
jerk_lat = [-4, 0, 4]
labels = [f"L{jl}/S{sl}" for jl in jerk_long for sl in jerk_lat]

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
counts = np.bincount(valid_actions.astype(int), minlength=12)
ax.bar(range(12), counts / counts.sum())
ax.set_xticks(range(12))
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel("Frequency")
ax.set_title("Action Distribution")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Episode Return vs Length

# %%
flat_rewards = rewards.reshape(-1)
flat_terminals = terminals.reshape(-1)
flat_truncations = truncations.reshape(-1)
flat_invalid_mask = is_invalid.reshape(-1)

done_mask = (flat_terminals + flat_truncations).clip(max=1)
valid_mask = flat_invalid_mask == 0

episode_ends = np.where(done_mask > 0)[0]
episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])

episode_returns = []
episode_lengths = []
for start, end in zip(episode_starts, episode_ends):
    ep_valid = valid_mask[start : end + 1]
    ep_return = flat_rewards[start : end + 1][ep_valid].sum()
    ep_length = ep_valid.sum()
    episode_returns.append(ep_return)
    episode_lengths.append(ep_length)

episode_returns = np.array(episode_returns)
episode_lengths = np.array(episode_lengths)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(episode_returns, bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Episode Return")
axes[0].set_title(f"Returns (mean={episode_returns.mean():.2f})")
axes[0].axvline(episode_returns.mean(), color="red", linestyle="--")

axes[1].hist(episode_lengths, bins=50, edgecolor="black", alpha=0.7, color="orange")
axes[1].set_xlabel("Episode Length")
axes[1].set_title(f"Lengths (mean={episode_lengths.mean():.1f})")

axes[2].scatter(episode_lengths, episode_returns, alpha=0.1, s=5)
axes[2].set_xlabel("Episode Length")
axes[2].set_ylabel("Episode Return")
axes[2].set_title("Return vs Length")

plt.tight_layout()
plt.show()

print(f"Episodes: {len(episode_returns)}")
print(f"Mean return: {episode_returns.mean():.2f} +/- {episode_returns.std():.2f}")
print(f"Mean length: {episode_lengths.mean():.1f}")

# %% [markdown]
# ## Spawn Position Analysis

# %%
valid = traj_lengths > 5
start_x = traj_x[valid, 0]
start_y = traj_y[valid, 0]
end_x = np.array([traj_x[i, traj_lengths[i] - 1] for i in np.where(valid)[0]])
end_y = np.array([traj_y[i, traj_lengths[i] - 1] for i in np.where(valid)[0]])

# Displacement
dist = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

# Total path length
path_lengths = []
for i in np.where(valid)[0]:
    length = traj_lengths[i]
    dx = np.diff(traj_x[i, :length])
    dy = np.diff(traj_y[i, :length])
    path_lengths.append(np.sqrt(dx**2 + dy**2).sum())
path_lengths = np.array(path_lengths)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(dist, bins=50, edgecolor="black", alpha=0.7, color="green")
axes[0].set_xlabel("Euclidean Distance Start->End (m)")
axes[0].set_title(f"Displacement (mean={dist.mean():.1f}m)")

axes[1].hist(path_lengths, bins=50, edgecolor="black", alpha=0.7, color="purple")
axes[1].set_xlabel("Total Path Length (m)")
axes[1].set_title(f"Path Length (mean={path_lengths.mean():.1f}m)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Interactive Map + Trajectory Viewer
#
# Load map binary files and overlay agent trajectories. Use the slider to select agents ranked by trajectory length.

# %%
import struct
import ipywidgets as widgets
from IPython.display import display

# Road type constants (from drive.h)
ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6
DRIVEWAY = 10


def load_map_roads(map_path):
    """Read road elements from a PufferDrive binary map file."""
    roads = []
    with open(map_path, "rb") as f:
        sdc_track_index = struct.unpack("i", f.read(4))[0]
        num_tracks_to_predict = struct.unpack("i", f.read(4))[0]
        if num_tracks_to_predict > 0:
            f.read(num_tracks_to_predict * 4)  # skip track indices

        num_objects = struct.unpack("i", f.read(4))[0]
        num_roads = struct.unpack("i", f.read(4))[0]

        total_entities = num_objects + num_roads
        for i in range(total_entities):
            scenario_id = struct.unpack("i", f.read(4))[0]
            entity_type = struct.unpack("i", f.read(4))[0]
            entity_id = struct.unpack("i", f.read(4))[0]
            array_size = struct.unpack("i", f.read(4))[0]

            if i < num_objects:
                # Agent: skip trajectory arrays + scalar fields
                # x, y, z, vx, vy, vz (6 float arrays) + heading (float) + valid (int)
                f.read(array_size * 4 * 6)  # 6 float arrays
                f.read(array_size * 4)  # heading (float)
                f.read(array_size * 4)  # valid (int)
                f.read(4 * 3 + 4 * 3 + 4)  # width,length,height + goal xyz + mark_as_expert
            else:
                # Road element
                x = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                y = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                z = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                f.read(4 * 3 + 4 * 3 + 4)  # skip scalar fields
                roads.append({"type": entity_type, "x": x, "y": y, "z": z})

    return roads


def mean_center_roads(roads, world_mean):
    """Subtract world_mean from road coordinates to match simulation frame."""
    for r in roads:
        r["x"] = r["x"] - world_mean[0]
        r["y"] = r["y"] - world_mean[1]
        if len(world_mean) > 2:
            r["z"] = r["z"] - world_mean[2]
    return roads


# Get world_mean from trajectory data (exact value from C code)
world_mean = data.get("world_mean", None)
if world_mean is not None:
    print(f"Using world_mean from trajectory data: {world_mean}")
else:
    print("WARNING: world_mean not in trajectory data, map alignment may be off")

# Load all maps — resolve relative paths against project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
map_roads = {}
for mid, mf in enumerate(map_files):
    mf_str = str(mf)
    mf_path = Path(mf_str)
    if not mf_path.exists():
        mf_path = PROJECT_ROOT / mf_str
    if mf_path.exists():
        roads = load_map_roads(str(mf_path))
        if world_mean is not None:
            mean_center_roads(roads, world_mean)
        map_roads[mid] = roads
        print(f"Map {mid} ({mf_path.stem}): {len(roads)} road elements")
    else:
        print(f"Map {mid} ({mf_str}): not found at {mf_path}, skipping")


# %%
from matplotlib.patches import FancyArrow
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def draw_map_background(ax, mid):
    """Draw road elements for a given map onto ax."""
    if mid not in map_roads:
        return
    for road in map_roads[mid]:
        if road["type"] == ROAD_EDGE:
            ax.plot(road["x"], road["y"], color="gray", linewidth=0.8, alpha=0.6)
        elif road["type"] == ROAD_LANE:
            ax.plot(road["x"], road["y"], color="khaki", linewidth=0.5, alpha=0.4)
        elif road["type"] == ROAD_LINE:
            ax.plot(road["x"], road["y"], color="white", linewidth=0.3, alpha=0.3)


def draw_map_with_trajectory(agent_idx):
    """Draw map roads and overlay the full trajectory for an agent."""
    mid = map_ids[agent_idx]
    length = traj_lengths[agent_idx]

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    draw_map_background(ax, mid)

    if length > 1:
        x = traj_x[agent_idx, :length]
        y = traj_y[agent_idx, :length]
        h = traj_heading[agent_idx, :length]

        sc = ax.scatter(x, y, c=np.arange(length), cmap="plasma", s=15, zorder=5, alpha=0.9)
        ax.plot(x[0], y[0], "go", markersize=12, zorder=6, label="start")
        ax.plot(x[-1], y[-1], "rx", markersize=12, zorder=6, label="end")

        arrow_step = max(1, length // 20)
        for t in range(0, length, arrow_step):
            dx = np.cos(h[t]) * 2
            dy = np.sin(h[t]) * 2
            ax.arrow(x[t], y[t], dx, dy, head_width=0.5, head_length=0.3, fc="cyan", ec="cyan", alpha=0.7, zorder=7)

        plt.colorbar(sc, ax=ax, label="Step", shrink=0.7)
        pad = 30
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(y.min() - pad, y.max() + pad)

    ax.set_aspect("equal")
    ax.set_facecolor("#2a2a2a")
    ax.set_title(
        f"Agent {agent_idx} | Map {mid} ({Path(str(map_files[mid])).stem}) | Length: {length} steps", fontsize=13
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if length > 1:
        ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def make_trajectory_video(agent_idx, follow_agent=True, window_size=60):
    """Create an animation of the agent's trajectory unrolling on the map."""
    mid = map_ids[agent_idx]
    length = traj_lengths[agent_idx]
    if length < 2:
        print(f"Agent {agent_idx} has no trajectory data")
        return None

    x = traj_x[agent_idx, :length]
    y = traj_y[agent_idx, :length]
    h = traj_heading[agent_idx, :length]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    draw_map_background(ax, mid)
    ax.set_aspect("equal")
    ax.set_facecolor("#2a2a2a")

    # Trail line (grows over time)
    (trail_line,) = ax.plot([], [], color="cyan", linewidth=2, alpha=0.6, zorder=4)
    # Current position marker
    (car_marker,) = ax.plot([], [], "o", color="lime", markersize=10, zorder=6)
    # Heading arrow (updated each frame)
    heading_arrow = None
    # Start marker
    ax.plot(x[0], y[0], "s", color="lime", markersize=8, zorder=5, label="start")
    title = ax.set_title("", fontsize=13)

    if follow_agent:
        half = window_size / 2
    else:
        pad = 30
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(y.min() - pad, y.max() + pad)

    def init():
        trail_line.set_data([], [])
        car_marker.set_data([], [])
        return trail_line, car_marker

    def animate(frame):
        nonlocal heading_arrow
        t = frame

        # Update trail
        trail_line.set_data(x[: t + 1], y[: t + 1])
        # Update car position
        car_marker.set_data([x[t]], [y[t]])

        # Update heading arrow
        if heading_arrow is not None:
            heading_arrow.remove()
        arrow_len = 3
        dx = np.cos(h[t]) * arrow_len
        dy = np.sin(h[t]) * arrow_len
        heading_arrow = ax.arrow(x[t], y[t], dx, dy, head_width=1.0, head_length=0.5, fc="red", ec="red", zorder=7)

        # Camera follow
        if follow_agent:
            ax.set_xlim(x[t] - half, x[t] + half)
            ax.set_ylim(y[t] - half, y[t] + half)

        # Speed from position diff
        if t > 0:
            spd = np.sqrt((x[t] - x[t - 1]) ** 2 + (y[t] - y[t - 1]) ** 2) / 0.1
        else:
            spd = 0
        title.set_text(f"Agent {agent_idx} | Step {t}/{length} | Speed: {spd:.1f} m/s")

        return trail_line, car_marker, heading_arrow

    anim = FuncAnimation(fig, animate, init_func=init, frames=length, interval=100, blit=False)
    plt.close(fig)
    return anim


# Build list of agents with trajectories, sorted by length (longest first)
agents_with_data = np.where(traj_lengths > 1)[0]
agents_sorted = agents_with_data[np.argsort(traj_lengths[agents_with_data])[::-1]]

print(f"{len(agents_sorted)} agents with trajectory data")
print(f"Longest: agent {agents_sorted[0]} with {traj_lengths[agents_sorted[0]]} steps")

# %% [markdown]
# ### Static view — select agent by rank (0 = longest trajectory)

# %%
output = widgets.Output()

agent_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=len(agents_sorted) - 1,
    step=1,
    description="Rank:",
    continuous_update=False,
    layout=widgets.Layout(width="80%"),
)
agent_label = widgets.Label(value="")


def update_static(change):
    idx = agents_sorted[change["new"]]
    agent_label.value = f"Agent {idx} | Map {map_ids[idx]} | Length {traj_lengths[idx]} steps"
    with output:
        output.clear_output(wait=True)
        draw_map_with_trajectory(idx)


agent_slider.observe(update_static, names="value")
display(
    widgets.VBox(
        [
            widgets.HTML("<h3>Static Trajectory View</h3>"),
            widgets.HBox([agent_slider, agent_label]),
            output,
        ]
    )
)
update_static({"new": 0})

# %% [markdown]
# ### Animated rollout — watch the agent drive
#
# Set `AGENT_RANK` below to pick which agent to animate (0 = longest trajectory).
# Set `FOLLOW = True` to have the camera follow the agent, `False` for fixed view.

# %%
AGENT_RANK = 800  # change this to pick a different agent
FOLLOW = True  # camera follows agent

agent_idx = agents_sorted[AGENT_RANK]
print(
    f"Animating agent {agent_idx} (rank {AGENT_RANK}), length {traj_lengths[agent_idx]} steps, map {map_ids[agent_idx]}"
)

anim = make_trajectory_video(agent_idx, follow_agent=FOLLOW, window_size=60)
if anim is not None:
    display(HTML(anim.to_jshtml()))

# %%
