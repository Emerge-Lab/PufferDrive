# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 - Observation Pipeline Debug
# Verify obs vector is correctly packed, normalized, interpretable.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pufferlib.viz import plot_observation, plot_simulator_state, unpack_obs
from notebooks.notebook_utils import COEF_NAMES, make_drive_env, zero_actions

env, obs, info = make_drive_env()

# %% [markdown]
# ## Raw obs inspection

# %%
# Take first step so obs are populated
actions = zero_actions(env)

obs, rew, term, trunc, info = env.step(actions)

print(f"shape: {obs.shape}, dtype: {obs.dtype}")
print(f"min: {obs.min():.4f}, max: {obs.max():.4f}, mean: {obs.mean():.4f}, std: {obs.std():.4f}")
print(f"NaN: {np.isnan(obs).sum()}, Inf: {np.isinf(obs).sum()}")
print(f"% zeros: {(obs == 0).mean() * 100:.1f}%")
print(f"% outside [-1,1]: {((obs < -1) | (obs > 1)).mean() * 100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(obs.flatten(), bins=100, edgecolor="black", alpha=0.7)
axes[0].set_title("Full obs distribution")
axes[0].set_xlabel("Value")
# Per-agent: show obs[0] vs obs[1]
for i in range(min(4, obs.shape[0])):
    axes[1].plot(obs[i], alpha=0.5, label=f"agent {i}")
axes[1].set_title("Obs vector by index (first 4 agents)")
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Unpack with pufferlib.viz.unpack_obs

# %%
ego, target, partners, lanes, boundaries, traffic = unpack_obs(
    obs[:1],
    target_type=env.target_type,
    reward_conditioning=env.reward_conditioning,
    num_target_waypoints=env.num_target_waypoints,
    obs_slots_partners_n=env.obs_slots_partners_n,
    obs_slots_lane_n=env.obs_slots_lane_kept,
    obs_slots_boundary_n=env.obs_slots_boundary_kept,
    obs_slots_traffic_controls_n=env.obs_slots_traffic_controls_n,
)
print(f"ego: {ego.shape} = {ego}")
print(f"target: {target.shape}")
print(f"partners: {partners.shape}")
print(f"lanes: {lanes.shape}")
print(f"boundaries: {boundaries.shape}")
print(f"traffic: {traffic.shape}")


labels = [
    "speed",
    "width",
    "length",
    "steering",
    "a_long",
    "a_lat",
    "lane_center_dist_01",
    "lane_heading_cos",
    "speed_limit",
]
for name, val in zip(labels, ego):
    print(f"  {name}: {val:.4f}")

# %% [markdown]
# ## Manual slice verification

# %%
o = obs[0]  # first agent flat obs
idx = 0

# Ego
ego_manual = o[idx : idx + env.ego_features]
idx += env.ego_features
assert np.allclose(ego_manual, ego), f"ego mismatch: {ego_manual} vs {ego}"

# Reward conditioning coefs
coefs_manual = o[idx : idx + env.num_reward_coefs]
idx += env.num_reward_coefs

# Target
target_manual = o[idx : idx + env.num_target_waypoints * env.target_features].reshape(
    env.num_target_waypoints, env.target_features
)
idx += env.num_target_waypoints * env.target_features
assert np.allclose(target_manual, target), "target mismatch"

# Partners
partners_manual = o[idx : idx + env.obs_slots_partners_n * env.partner_features].reshape(
    env.obs_slots_partners_n, env.partner_features
)
idx += env.obs_slots_partners_n * env.partner_features
assert np.allclose(partners_manual, partners), "partners mismatch"

# Lanes
lanes_manual = o[idx : idx + env.obs_slots_lane_kept * env.road_features].reshape(
    env.obs_slots_lane_kept, env.road_features
)
idx += env.obs_slots_lane_kept * env.road_features
assert np.allclose(lanes_manual, lanes), "lanes mismatch"

# Boundaries
bounds_manual = o[idx : idx + env.obs_slots_boundary_kept * env.road_features].reshape(
    env.obs_slots_boundary_kept, env.road_features
)
idx += env.obs_slots_boundary_kept * env.road_features
assert np.allclose(bounds_manual, boundaries), "boundaries mismatch"

# Traffic
traffic_manual = o[idx : idx + env.obs_slots_traffic_controls_n * env.traffic_control_features].reshape(
    env.obs_slots_traffic_controls_n, env.traffic_control_features
)
idx += env.obs_slots_traffic_controls_n * env.traffic_control_features
assert np.allclose(traffic_manual, traffic), "traffic mismatch"

assert idx == obs.shape[1], f"obs size mismatch: used {idx}, total {obs.shape[1]}"
print(f"All slices match. Total features used: {idx}")

# %% [markdown]
# ## Reward conditioning coefficients

# %%
coefs = obs[0, env.ego_features : env.ego_features + env.num_reward_coefs]
fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(range(env.num_reward_coefs), coefs, tick_label=COEF_NAMES)
ax.set_ylabel("Normalized coef value")
ax.set_title("Reward conditioning coefficients (agent 0)")
plt.xticks(rotation=45, ha="right")
for bar, val in zip(bars, coefs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.show()

# Compare across agents
all_coefs = obs[:, env.ego_features : env.ego_features + env.num_reward_coefs]
print("Coef stats across agents:")
for i, name in enumerate(COEF_NAMES):
    c = all_coefs[:, i]
    print(f"  {name:15s}: mean={c.mean():.3f} std={c.std():.3f} min={c.min():.3f} max={c.max():.3f}")

# %% [markdown]
# ## Partner observations

# %%
partner_labels = [
    "rel_x",
    "rel_y",
    "rel_z",
    "length",
    "width",
    "heading_cos",
    "heading_sin",
    "speed",
    "seconds_stopped",
]
active_mask = ~np.all(partners == 0, axis=1)
n_active = active_mask.sum()
print(f"Active partners: {n_active}/{env.obs_slots_partners_n}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
im = axes[0].imshow(partners, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_xticks(range(env.partner_features))
axes[0].set_xticklabels(partner_labels, rotation=45, ha="right")
axes[0].set_ylabel("Partner index")
axes[0].set_title(f"Partner obs heatmap ({n_active} active)")
plt.colorbar(im, ax=axes[0])

# Scatter in ego frame
active_partners = partners[active_mask]
if len(active_partners) > 0:
    axes[1].scatter(active_partners[:, 0], active_partners[:, 1], c="gray", s=100, edgecolors="black")
    for i, p in enumerate(active_partners):
        axes[1].annotate(str(i), (p[0], p[1]), fontsize=8, ha="center", va="bottom")
axes[1].scatter(0, 0, c="blue", s=200, marker="s", label="ego", zorder=10)
axes[1].set_xlabel("rel_x")
axes[1].set_ylabel("rel_y")
axes[1].set_title("Partners in ego frame")
axes[1].legend()
axes[1].set_aspect("equal")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Lane / boundary segments

# %%
road_labels = ["rel_x", "rel_y", "rel_z", "length", "width", "dir_cos", "dir_sin"]

lane_active = ~np.all(lanes == 0, axis=1)
bound_active = ~np.all(boundaries == 0, axis=1)
print(
    f"Active lanes: {lane_active.sum()}/{env.obs_slots_lane_kept}, boundaries: {bound_active.sum()}/{env.obs_slots_boundary_kept}"
)

fig, ax = plt.subplots(figsize=(10, 10))

# Mirror the canonical road rendering in pufferlib.viz.plot_observation
for seg in lanes[lane_active]:
    x, y, z, length, width, dc, ds = seg
    ax.scatter(x, y, color="lightgrey", s=10, zorder=1)
    ax.plot(
        [x + dc * length / 2, x - dc * length / 2],
        [y + ds * length / 2, y - ds * length / 2],
        color="lightgrey",
        linewidth=1,
        zorder=1,
    )

for seg in boundaries[bound_active]:
    x, y, z, length, width, dc, ds = seg
    ax.scatter(x, y, color="black", s=10, zorder=1)
    ax.plot(
        [x + dc * length / 2, x - dc * length / 2],
        [y + ds * length / 2, y - ds * length / 2],
        color="black",
        linewidth=1,
        zorder=1,
    )

ax.scatter(0, 0, color="blue", s=200, marker="s", label="ego", zorder=10)
ax.text(
    0.12,
    0.95,
    f"Lanes: {lane_active.sum()}\nBoundaries: {bound_active.sum()}",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)
ax.axis((-1, 1, -1, 1))
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("X (ego frame)")
ax.set_ylabel("Y (ego frame)")
ax.set_title("Lane + boundary segments in ego frame")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Ego-centric view (pufferlib.viz)

# %%
img = plot_observation(
    obs[:1],
    target_type=env.target_type,
    reward_conditioning=env.reward_conditioning,
    num_target_waypoints=env.num_target_waypoints,
    obs_slots_partners_n=env.obs_slots_partners_n,
    obs_slots_lane_n=env.obs_slots_lane_kept,
    obs_slots_boundary_n=env.obs_slots_boundary_kept,
    obs_slots_traffic_controls_n=env.obs_slots_traffic_controls_n,
)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
ax.axis("off")
ax.set_title("Ego-centric observation (agent 0)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Bird's eye view (simulator state)

# %%
scenarios = env.get_state()
# get_state returns a list of scenario dicts (one per sub-env) or a single dict
if isinstance(scenarios, list):
    scenario = scenarios[0]
else:
    scenario = scenarios

img = plot_simulator_state(scenario)
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(img)
ax.axis("off")
ax.set_title("Bird's eye view")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Multi-step: ego features over time

# %%
N_STEPS = 20
ego_history = np.zeros((N_STEPS, env.ego_features))

for t in range(N_STEPS):
    actions = zero_actions(env)
    obs_t, _, _, _, _ = env.step(actions)
    ego_history[t] = obs_t[0, : env.ego_features]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
# Speed
axes[0, 0].plot(ego_history[:, 0])
axes[0, 0].set_title("speed")
axes[0, 0].set_xlabel("step")
# Steering
axes[0, 1].plot(ego_history[:, 3])
axes[0, 1].set_title("steering")
axes[0, 1].set_xlabel("step")
# a_long
axes[1, 0].plot(ego_history[:, 4])
axes[1, 0].set_title("a_long")
axes[1, 0].set_xlabel("step")
# a_lat
axes[1, 1].plot(ego_history[:, 5])
axes[1, 1].set_title("a_lat")
axes[1, 1].set_xlabel("step")
plt.suptitle("Agent 0 ego features over 20 steps (no-op action)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Cross-agent distributions

# %%
# Current obs across all agents
# Ego features (jerk): speed(0), width(1), length(2), steering(3), a_long(4), a_lat(5), lane_center(6), lane_heading(7), speed_limit(8)
speeds = obs[:, 0]  # speed is at index 0

# Target waypoints start after ego + reward coefs
target_start = env.ego_features + env.num_reward_coefs
# Each target waypoint has TARGET_F features; first two are rel_x, rel_y
first_target_x = obs[:, target_start]
first_target_y = obs[:, target_start + 1]
target_dists = np.sqrt(first_target_x**2 + first_target_y**2)

# Count active partners per agent
partner_start = env.ego_features + env.num_reward_coefs + env.num_target_waypoints * env.target_features
partner_end = partner_start + env.obs_slots_partners_n * env.partner_features
all_partners = obs[:, partner_start:partner_end].reshape(-1, env.obs_slots_partners_n, env.partner_features)
partner_counts = (~np.all(all_partners == 0, axis=2)).sum(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(speeds, bins=20, edgecolor="black", alpha=0.7)
axes[0].set_title(f"Speed distribution (N={len(speeds)})")
axes[0].set_xlabel("speed")

axes[1].hist(target_dists, bins=20, edgecolor="black", alpha=0.7, color="orange")
axes[1].set_title("Distance to first target waypoint")
axes[1].set_xlabel("distance")

axes[2].hist(partner_counts, bins=range(env.obs_slots_partners_n + 2), edgecolor="black", alpha=0.7, color="green")
axes[2].set_title("Active partners per agent")
axes[2].set_xlabel("count")
plt.tight_layout()
plt.show()
