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
# # 05 - Model Inference Debug
# End-to-end inference pipeline: config loading, policy forward pass, rollouts (deterministic vs stochastic), observation/reward analysis, value accuracy, trajectories, LSTM state.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import binding
from pufferlib.ocean.torch import Drive as DrivePolicy
from pufferlib.pytorch import sample_logits
from notebooks.notebook_utils import COEF_NAMES, EGO_LABELS, MAP_DIR, load_notebook_config, zero_actions

CHECKPOINT_PATH = ""
ENV_NAME = "puffer_drive"

config = load_notebook_config(CHECKPOINT_PATH, ENV_NAME)
config["env"]["num_agents"] = 64
config["env"]["num_maps"] = 8
config["env"]["eval_mode"] = 1
config["env"]["map_dir"] = MAP_DIR

config["env"]["obs_slots_boundary_n"] = 80
config["env"]["obs_slots_lane_n"] = 80
config["env"]["obs_dropout_lane"] = 0.0
config["env"]["obs_dropout_boundary"] = 0.0

env = Drive(**config["env"])
obs, info = env.reset(seed=42)
N = env.num_agents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = DrivePolicy(env, **config["policy"]).to(device)

if CHECKPOINT_PATH:
    sd = torch.load(CHECKPOINT_PATH, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    policy.load_state_dict(sd)
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

is_continuous = policy.is_continuous
ACT_SHAPE = (N, len(env.single_action_space.nvec)) if not is_continuous else (N, env.single_action_space.shape[0])

print(f"Policy on {device}, params: {sum(p.numel() for p in policy.parameters()):,}")
print(f"Obs shape: {obs.shape}, Action space: {env.single_action_space}")
print(f"Config: dynamics={config['env']['dynamics_model']}, action={config['env']['action_type']}")

# %% [markdown]
# ## Single-step policy output

# %%
# Take one step to get fresh obs
actions = zero_actions(env)
obs, rew, term, trunc, info = env.step(actions)

obs_tensor = torch.FloatTensor(obs).to(device)
policy.eval()

with torch.no_grad():
    logits_list, value = policy(obs_tensor)

# Sample actions
action, logprob, ent, _ = sample_logits(logits_list)
action_det, _, _, _ = sample_logits(logits_list, deterministic=True) # TODO

print(f"Value: mean={value.mean():.4f}, std={value.std():.4f}, range=[{value.min():.4f}, {value.max():.4f}]")
print(f"Entropy: mean={ent.mean():.4f}, std={ent.std():.4f}")
print(f"LogProb: mean={logprob.mean():.4f}, std={logprob.std():.4f}")
print(f"Stochastic action sample: {action[0].cpu().numpy()}")
print(f"Deterministic action: {action_det[0].cpu().numpy()}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Action probs (first head for multi-discrete, or full logits)
if isinstance(logits_list, list) or isinstance(logits_list, tuple):
    probs = F.softmax(logits_list[0], dim=-1)
else:
    probs = F.softmax(logits_list, dim=-1)
mean_probs = probs.mean(dim=0).cpu().numpy()
axes[0].bar(range(len(mean_probs)), mean_probs, edgecolor="black", alpha=0.7)
axes[0].axhline(1.0 / len(mean_probs), color="red", ls="--", label="uniform")
axes[0].set_xlabel("Action")
axes[0].set_ylabel("Probability")
axes[0].set_title("Mean action probabilities (across agents)")
axes[0].legend()

axes[1].hist(value.cpu().numpy().flatten(), bins=30, edgecolor="black", alpha=0.7, color="purple")
axes[1].set_title("Value predictions across agents")
axes[1].set_xlabel("Value")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Full rollout: deterministic vs stochastic

# %%
HORIZON = 256
TRACKED_AGENT = 0  # agent index to track in detail
obs_dim = obs.shape[1]

dyn_model = config["env"]["dynamics_model"]
tgt_type = config["env"]["target_type"]
rew_cond = config["env"].get("reward_conditioning", False)
n_tgt_wp = config["env"].get("num_target_waypoints", 3)


def run_rollout(env, policy, deterministic=False, horizon=HORIZON):
    obs, _ = env.reset(seed=42)
    N = env.num_agents

    buffers = {
        "obs": np.zeros((horizon, N, obs_dim), dtype=np.float32),
        "actions": np.zeros((horizon, N), dtype=np.int64),
        "rewards": np.zeros((horizon, N), dtype=np.float32),
        "values": np.zeros((horizon, N), dtype=np.float32),
        "logprobs": np.zeros((horizon, N), dtype=np.float32),
        "entropy": np.zeros((horizon, N), dtype=np.float32),
        "terminals": np.zeros((horizon, N), dtype=np.float32),
        "truncations": np.zeros((horizon, N), dtype=np.float32),
        "positions_x": np.zeros((horizon, N), dtype=np.float32),
        "positions_y": np.zeros((horizon, N), dtype=np.float32),
    }

    policy.eval()
    for t in range(horizon):
        obs_t = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            logits_list, val = policy(obs_t)
            act, logp, entr, _ = sample_logits(logits_list, deterministic=deterministic)

        buffers["obs"][t] = obs
        buffers["actions"][t] = act.cpu().numpy().reshape(N) if act.dim() > 1 else act.cpu().numpy()
        buffers["values"][t] = val.squeeze().cpu().numpy()
        buffers["logprobs"][t] = logp.cpu().numpy()
        buffers["entropy"][t] = entr.cpu().numpy()

        # Get positions
        gstate = env.get_global_agent_state()
        buffers["positions_x"][t] = gstate["x"]
        buffers["positions_y"][t] = gstate["y"]

        # Step env
        env_actions = act.cpu().numpy().reshape(ACT_SHAPE)
        obs, rew, term, trunc, info = env.step(env_actions)
        buffers["rewards"][t] = rew
        buffers["terminals"][t] = term
        buffers["truncations"][t] = trunc

    return buffers


print("Running stochastic rollout...")
buf_stoch = run_rollout(env, policy, deterministic=False)
print("Running deterministic rollout...")
buf_det = run_rollout(env, policy, deterministic=True)

for name, buf in [("Stochastic", buf_stoch), ("Deterministic", buf_det)]:
    print(f"\n--- {name} ---")
    print(f"  Reward: mean={buf['rewards'].mean():.5f}, std={buf['rewards'].std():.5f}")
    print(f"  Value: mean={buf['values'].mean():.5f}, std={buf['values'].std():.5f}")
    print(f"  Entropy: mean={buf['entropy'].mean():.4f}")
    print(f"  Terminals: {buf['terminals'].sum():.0f}, Truncations: {buf['truncations'].sum():.0f}")

# %% [markdown]
# ## Observation analysis

# %%
from pufferlib.viz import unpack_obs, plot_observation, plot_simulator_state

# Ego-centric observation at t=50 for tracked agent
sample_t = min(50, HORIZON - 1)
sample_obs = buf_stoch["obs"][sample_t : sample_t + 1, TRACKED_AGENT : TRACKED_AGENT + 1][0]
print(dyn_model, tgt_type, rew_cond, n_tgt_wp)
img = plot_observation(
    sample_obs,
    target_type=tgt_type,
    reward_conditioning=rew_cond,
    num_target_waypoints=n_tgt_wp,
    obs_slots_partners_n=env.obs_slots_partners_n,
    obs_slots_lane_n=env.obs_slots_lane_n,
    obs_slots_boundary_n=env.obs_slots_boundary_n,
    obs_dropout_lane=env.obs_dropout_lane,
    obs_dropout_boundary=env.obs_dropout_boundary,
    obs_lane_stride=env.obs_lane_stride,
    obs_boundary_stride=env.obs_boundary_stride,
    obs_slots_traffic_controls_n=env.obs_slots_traffic_controls_n,
    obs_norm_goal_offset_m=env.obs_norm_goal_offset_m,
    obs_norm_xy_offset_m=env.obs_norm_xy_offset_m,
    obs_norm_veh_width_m=env.obs_norm_veh_width_m,
    obs_norm_veh_length_m=env.obs_norm_veh_length_m,
)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.title(f"Ego-centric obs | agent={TRACKED_AGENT}, t={sample_t}")
plt.show()

# BEV simulator state
scenarios = env.get_state()
if scenarios and len(scenarios) > 0:
    img_bev = plot_simulator_state(scenarios[0], timestep=0)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_bev)
    plt.axis("off")
    plt.title("BEV Simulator State")
    plt.show()

# Ego feature time series for tracked agent
ego_features_over_time = []
for t in range(HORIZON):
    ego, *_ = unpack_obs(
        buf_stoch["obs"][t : t + 1, TRACKED_AGENT : TRACKED_AGENT + 1][0],
        target_type=tgt_type,
        reward_conditioning=rew_cond,
        num_target_waypoints=n_tgt_wp,
        obs_slots_partners_n=env.obs_slots_partners_n,
        obs_slots_lane_n=env.obs_slots_lane_n,
        obs_slots_boundary_n=env.obs_slots_boundary_n,
        obs_dropout_lane=env.obs_dropout_lane,
        obs_dropout_boundary=env.obs_dropout_boundary,
        obs_slots_traffic_controls_n=env.obs_slots_traffic_controls_n,
    )
    ego_features_over_time.append(ego)
ego_ts = np.array(ego_features_over_time)

if dyn_model == "jerk":
    labels = ["speed", "width", "length", "steering", "accel_long", "accel_lat", "lcenter", "lalign", "speed_limit"]
    plot_idxs = [0, 3, 4, 5]  # speed, steering, accel_long, accel_lat
else:
    labels = ["speed", "width", "length", "lcenter", "lalign", "speed_limit"]
    plot_idxs = [0, 3, 4, 5]  # speed, lcenter, lalign, speed_limit

fig, axes = plt.subplots(len(plot_idxs), 1, figsize=(14, 3 * len(plot_idxs)), sharex=True)
for i, idx in enumerate(plot_idxs):
    axes[i].plot(ego_ts[:, idx])
    print(ego_ts[10:, idx].argmin())
    axes[i].set_ylabel(labels[idx])
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel("Step")
fig.suptitle(f"Ego features over time | agent={TRACKED_AGENT}", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Observation layer breakdown
#
# Obs layout (all ego-centric, normalized):
# - **Ego**: speed, width, length, [jerk: steering, accel_long, accel_lat], lane_center_dist, lane_angle, speed_limit
# - **Conditioning** (if enabled): 17 reward coefs (goal_radius, goal_speed, collision, offroad, comfort, lane_align, vel_align, lane_center, center_bias, velocity, reverse, stop_line, timestep, overspeed, throttle, steer, acc) + target waypoints
# - **Target**: static=rel_x,rel_y,rel_z per waypoint; dynamic=rel_x,rel_y,rel_z,heading_cos,heading_sin per waypoint
# - **Partners** (MAX_PARTNERS x 9): rel_x, rel_y, rel_z, length, width, heading_cos, heading_sin, sim_speed_signed, seconds_stopped
# - **Lanes** (MAX_LANES x 7): rel_x, rel_y, rel_z, seg_length, seg_width, dir_cos, dir_sin
# - **Boundaries** (MAX_BOUNDS x 7): same as lanes
# - **Traffic controls** (MAX_TRAFFIC x 7): rel_x1, rel_y1, rel_x2, rel_y2, rel_z, type, state

# %%
from pufferlib.viz import unpack_obs

sample_t = min(50, HORIZON - 1)
sample_obs = buf_stoch["obs"][sample_t : sample_t + 1, TRACKED_AGENT : TRACKED_AGENT + 1][0]
ego, target, partners, lanes, boundaries, traffic_controls = unpack_obs(
    sample_obs,
    target_type=tgt_type,
    reward_conditioning=rew_cond,
    num_target_waypoints=n_tgt_wp,
    obs_slots_partners_n=env.obs_slots_partners_n,
    obs_slots_lane_n=env.obs_slots_lane_n,
    obs_slots_boundary_n=env.obs_slots_boundary_n,
    obs_dropout_lane=env.obs_dropout_lane,
    obs_dropout_boundary=env.obs_dropout_boundary,
    obs_slots_traffic_controls_n=env.obs_slots_traffic_controls_n,
)

# Also unpack conditioning manually (unpack_obs doesn't return it separately)
ego_dim = binding.EGO_FEATURES
cond_dim = binding.NUM_REWARD_COEFS if rew_cond else 0
cond_obs = sample_obs[0, ego_dim : ego_dim + cond_dim] if cond_dim > 0 else None


# --- Print all layer shapes + stats ---
def layer_stats(name, arr):
    flat = arr.flatten() if hasattr(arr, "flatten") else np.array(arr).flatten()
    if flat.size == 0:
        print(f"{name:>14s}: shape={str(list(arr.shape)):>16s}  (empty)")
        return
    nonzero = np.count_nonzero(flat)
    print(
        f"{name:>14s}: shape={str(list(arr.shape)):>16s}  "
        f"nonzero={nonzero:>5d}/{flat.size:<5d}  "
        f"range=[{flat.min():.4f}, {flat.max():.4f}]  "
        f"mean={flat.mean():.4f}  std={flat.std():.4f}"
    )


print(f"--- Observation breakdown at t={sample_t}, agent={TRACKED_AGENT} ---")
print(f"Total obs dim: {sample_obs.shape[-1]}")
print()
layer_stats("Ego", ego)
if cond_obs is not None:
    layer_stats("Conditioning", cond_obs)
layer_stats("Target", target)
layer_stats("Partners", partners)
layer_stats("Lanes", lanes)
layer_stats("Boundaries", boundaries)
layer_stats("TrafficControls", traffic_controls)

# --- Ego features detail ---
ego_labels = EGO_LABELS

print(f"\n--- Ego features ---")
for i, (label, val) in enumerate(zip(ego_labels, ego)):
    print(f"  [{i}] {label:>14s} = {val:.4f}")

# --- Conditioning detail ---
if cond_obs is not None:
    cond_labels = COEF_NAMES
    print(f"\n--- Conditioning (reward coefs, normalized) ---")
    for i, (label, val) in enumerate(zip(cond_labels, cond_obs)):
        print(f"  [{i:>2d}] {label:>16s} = {val:.4f}")

# --- Target waypoints ---
tgt_feat = binding.STATIC_TARGET_FEATURES if tgt_type == "static" else binding.DYNAMIC_TARGET_FEATURES
if tgt_type == "static":
    tgt_labels = ["rel_x", "rel_y", "rel_z"]
else:
    tgt_labels = ["rel_x", "rel_y", "rel_z", "heading_cos", "heading_sin"]

print(f"\n--- Target waypoints (n={n_tgt_wp}, type={tgt_type}) ---")
for wp in range(target.shape[0]):
    vals = ", ".join(f"{tgt_labels[j]}={target[wp, j]:.4f}" for j in range(tgt_feat))
    active = "ACTIVE" if not np.allclose(target[wp], 0) else "zeroed"
    print(f"  wp[{wp}]: {vals}  ({active})")

# --- Partner summary ---
n_visible = np.sum(np.any(partners != 0, axis=1))
print(f"\n--- Partners: {n_visible}/{partners.shape[0]} visible ---")
partner_labels = [
    "rel_x",
    "rel_y",
    "rel_z",
    "length",
    "width",
    "heading_cos",
    "heading_sin",
    "sim_speed_signed",
    "seconds_stopped",
]
for p in range(min(int(n_visible), 5)):
    vals = ", ".join(f"{partner_labels[j]}={partners[p, j]:.3f}" for j in range(env.partner_features))
    print(f"  [{p}] {vals}")
if n_visible > 5:
    print(f"  ... ({n_visible - 5} more)")

# --- Lane/boundary occupancy ---
n_lanes = np.sum(np.any(lanes != 0, axis=1))
n_bounds = np.sum(np.any(boundaries != 0, axis=1))
print(f"\n--- Road: {n_lanes}/{lanes.shape[0]} lane segs, {n_bounds}/{boundaries.shape[0]} boundary segs ---")

# --- Traffic ---
n_traffic = np.sum(np.any(traffic_controls != 0, axis=1))
print(f"\n--- Traffic controls: {n_traffic}/{traffic_controls.shape[0]} visible ---")
traffic_labels = ["rel_x1", "rel_y1", "rel_x2", "rel_y2", "rel_z", "type", "state"]
for t in range(min(int(n_traffic), 5)):
    vals = ", ".join(
        f"{traffic_labels[j]}={traffic_controls[t, j]:.3f}"
        for j in range(min(len(traffic_labels), traffic_controls.shape[1]))
    )
    print(f"  [{t}] {vals}")

# %%
# --- Layer-level stats across ALL agents at sample_t ---
all_obs = buf_stoch["obs"][sample_t]  # (N, obs_dim)

ego_dim = binding.EGO_FEATURES
cond_dim = binding.NUM_REWARD_COEFS if rew_cond else 0
tgt_feat = binding.STATIC_TARGET_FEATURES if tgt_type == "static" else binding.DYNAMIC_TARGET_FEATURES
tgt_dim = n_tgt_wp * tgt_feat
partner_dim = env.obs_slots_partners_n * env.partner_features
lane_dim = env.obs_slots_lane_kept * env.road_features
boundary_dim = env.obs_slots_boundary_kept * env.road_features
traffic_dim = env.obs_slots_traffic_controls_n * env.traffic_control_features

# Slice indices
idx = 0
slices = {}
slices["ego"] = (idx, idx + ego_dim)
idx += ego_dim
if cond_dim > 0:
    slices["conditioning"] = (idx, idx + cond_dim)
    idx += cond_dim
slices["target"] = (idx, idx + tgt_dim)
idx += tgt_dim
slices["partners"] = (idx, idx + partner_dim)
idx += partner_dim
slices["lanes"] = (idx, idx + lane_dim)
idx += lane_dim
slices["boundaries"] = (idx, idx + boundary_dim)
idx += boundary_dim
slices["traffic"] = (idx, idx + traffic_dim)
idx += traffic_dim

print(f"Obs dim used: {idx} / {all_obs.shape[1]}")
print(
    f"\n{'Layer':>14s} | {'start':>5s}-{'end':>5s} | {'dim':>5s} | {'mean':>8s} | {'std':>8s} | {'min':>8s} | {'max':>8s} | {'%nonzero':>8s}"
)
print("-" * 95)
for name, (s, e) in slices.items():
    chunk = all_obs[:, s:e]
    nz_pct = 100 * np.count_nonzero(chunk) / chunk.size
    print(
        f"{name:>14s} | {s:>5d}-{e:>5d} | {e - s:>5d} | {chunk.mean():>8.4f} | {chunk.std():>8.4f} | "
        f"{chunk.min():>8.4f} | {chunk.max():>8.4f} | {nz_pct:>7.1f}%"
    )

# --- Plots ---
n_layers = len(slices)
fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(5 * ((n_layers + 1) // 2), 8))
axes = axes.flatten()

for i, (name, (s, e)) in enumerate(slices.items()):
    chunk = all_obs[:, s:e].flatten()
    # Filter out exact zeros for histogram readability on sparse layers
    nonzero_vals = chunk[chunk != 0]
    if len(nonzero_vals) > 0:
        axes[i].hist(nonzero_vals, bins=50, edgecolor="black", alpha=0.7)
        axes[i].set_title(f"{name} (nonzero only, {len(nonzero_vals)}/{len(chunk)})")
    else:
        axes[i].hist(chunk, bins=50, edgecolor="black", alpha=0.7)
        axes[i].set_title(f"{name} (all zeros)")
    axes[i].set_xlabel("Value")

# Hide unused axes
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(f"Observation distributions across {N} agents at t={sample_t}", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# --- Per-feature detail for partners, lanes, boundaries over time (tracked agent) ---


def unpack_all_timesteps(bufs, agent_idx):
    """Unpack all obs layers across time for one agent."""
    H = bufs["obs"].shape[0]
    egos, targets, conds = [], [], []
    n_partners, n_lanes, n_bounds, n_traffic = [], [], [], []

    for t in range(H):
        ob = bufs["obs"][t : t + 1, agent_idx : agent_idx + 1][0]
        ego, tgt, part, lane, bnd, tfc = unpack_obs(
            ob,
            target_type=tgt_type,
            reward_conditioning=rew_cond,
            num_target_waypoints=n_tgt_wp,
            obs_slots_partners_n=env.obs_slots_partners_n,
            obs_slots_lane_n=env.obs_slots_lane_n,
            obs_slots_boundary_n=env.obs_slots_boundary_n,
            obs_dropout_lane=env.obs_dropout_lane,
            obs_dropout_boundary=env.obs_dropout_boundary,
            obs_slots_traffic_controls_n=env.obs_slots_traffic_controls_n,
        )
        egos.append(ego)
        targets.append(tgt)
        n_partners.append(np.sum(np.any(part != 0, axis=1)))
        n_lanes.append(np.sum(np.any(lane != 0, axis=1)))
        n_bounds.append(np.sum(np.any(bnd != 0, axis=1)))
        n_traffic.append(np.sum(np.any(tfc != 0, axis=1)))

        if rew_cond:
            ed = binding.EGO_FEATURES
            conds.append(ob[0, ed : ed + binding.NUM_REWARD_COEFS])

    return {
        "ego": np.array(egos),
        "target": np.array(targets),
        "cond": np.array(conds) if conds else None,
        "n_partners": np.array(n_partners),
        "n_lanes": np.array(n_lanes),
        "n_bounds": np.array(n_bounds),
        "n_traffic": np.array(n_traffic),
    }


ts = unpack_all_timesteps(buf_stoch, TRACKED_AGENT)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Occupancy over time
axes[0, 0].plot(ts["n_partners"], label="partners", alpha=0.8)
axes[0, 0].plot(ts["n_lanes"], label="lanes", alpha=0.8)
axes[0, 0].plot(ts["n_bounds"], label="boundaries", alpha=0.8)
axes[0, 0].plot(ts["n_traffic"], label="traffic", alpha=0.8)
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Visible count")
axes[0, 0].set_title(f"Obs occupancy over time | agent={TRACKED_AGENT}")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Target waypoint distances over time
tgt_x = ts["target"][:, :, 0]
tgt_y = ts["target"][:, :, 1]
tgt_dist = np.sqrt(tgt_x**2 + tgt_y**2)
for wp in range(n_tgt_wp):
    axes[0, 1].plot(tgt_dist[:, wp], label=f"wp[{wp}]", alpha=0.8)
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Distance (normalized)")
axes[0, 1].set_title("Target waypoint distance over time")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Conditioning heatmap over time
if ts["cond"] is not None:
    cond_labels = COEF_NAMES
    im = axes[1, 0].imshow(ts["cond"].T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axes[1, 0].set_yticks(range(len(cond_labels)))
    axes[1, 0].set_yticklabels(cond_labels, fontsize=8)
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_title("Conditioning coefs over time")
    plt.colorbar(im, ax=axes[1, 0])
else:
    axes[1, 0].text(0.5, 0.5, "No conditioning", ha="center", va="center", transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("Conditioning (disabled)")

# Partner closest distance over time
partner_dists = []
for t in range(HORIZON):
    ob = buf_stoch["obs"][t : t + 1, TRACKED_AGENT : TRACKED_AGENT + 1][0]
    _, _, part, _, _, _ = unpack_obs(
        ob,
        target_type=tgt_type,
        reward_conditioning=rew_cond,
        num_target_waypoints=n_tgt_wp,
        obs_slots_partners_n=env.obs_slots_partners_n,
        obs_slots_lane_n=env.obs_slots_lane_n,
        obs_slots_boundary_n=env.obs_slots_boundary_n,
        obs_dropout_lane=env.obs_dropout_lane,
        obs_dropout_boundary=env.obs_dropout_boundary,
        obs_slots_traffic_controls_n=env.obs_slots_traffic_controls_n,
    )
    dists = np.sqrt(part[:, 0] ** 2 + part[:, 1] ** 2)
    visible = np.any(part != 0, axis=1)
    partner_dists.append(dists[visible].min() if visible.any() else np.nan)

axes[1, 1].plot(partner_dists, alpha=0.8, color="red")
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_ylabel("Min partner dist (normalized)")
axes[1, 1].set_title("Closest partner distance over time")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# --- Spatial scatter: all observed entities in ego frame at sample_t ---
sample_obs = buf_stoch["obs"][sample_t : sample_t + 1, TRACKED_AGENT : TRACKED_AGENT + 1][0]
ego, target, partners, lanes, boundaries, traffic_controls = unpack_obs(
    sample_obs,
    target_type=tgt_type,
    reward_conditioning=rew_cond,
    num_target_waypoints=n_tgt_wp,
    obs_slots_partners_n=env.obs_slots_partners_n,
    obs_slots_lane_n=env.obs_slots_lane_n,
    obs_slots_boundary_n=env.obs_slots_boundary_n,
    obs_dropout_lane=env.obs_dropout_lane,
    obs_dropout_boundary=env.obs_dropout_boundary,
    obs_slots_traffic_controls_n=env.obs_slots_traffic_controls_n,
)

fig, ax = plt.subplots(figsize=(10, 10))

# Ego vehicle at origin
from matplotlib.patches import Rectangle

ax.add_patch(
    Rectangle((-ego[2] / 2, -ego[1] / 2), ego[2], ego[1], facecolor="blue", edgecolor="black", alpha=0.7, zorder=10)
)
ax.annotate("EGO", (0, 0), fontsize=9, ha="center", va="center", color="white", fontweight="bold", zorder=11)

# Lane segments
for i in range(lanes.shape[0]):
    if np.allclose(lanes[i], 0):
        continue
    rx, ry, rz, length, _, dc, ds = lanes[i]
    ax.plot(
        [rx - dc * length / 2, rx + dc * length / 2],
        [ry - ds * length / 2, ry + ds * length / 2],
        color="lightgray",
        linewidth=1,
        zorder=1,
    )
ax.scatter(
    lanes[np.any(lanes != 0, axis=1), 0],
    lanes[np.any(lanes != 0, axis=1), 1],
    s=5,
    color="gray",
    alpha=0.5,
    label=f"lanes ({n_lanes})",
    zorder=2,
)

# Boundary segments
for i in range(boundaries.shape[0]):
    if np.allclose(boundaries[i], 0):
        continue
    rx, ry, rz, length, _, dc, ds = boundaries[i]
    ax.plot(
        [rx - dc * length / 2, rx + dc * length / 2],
        [ry - ds * length / 2, ry + ds * length / 2],
        color="black",
        linewidth=1,
        zorder=1,
    )
bnd_mask = np.any(boundaries != 0, axis=1)
if bnd_mask.any():
    ax.scatter(
        boundaries[bnd_mask, 0],
        boundaries[bnd_mask, 1],
        s=8,
        color="black",
        alpha=0.6,
        label=f"boundaries ({n_bounds})",
        zorder=2,
    )

# Partners
for i in range(partners.shape[0]):
    if np.allclose(partners[i], 0):
        continue
    rx, ry, rz, length, width, hc, hs, speed, _ = partners[i]
    heading = np.arctan2(hs, hc)
    rect = Rectangle(
        (-length / 2, -width / 2), length, width, facecolor="orange", edgecolor="black", alpha=0.6, zorder=9
    )
    rect.set_transform(plt.matplotlib.transforms.Affine2D().rotate(heading).translate(rx, ry) + ax.transData)
    ax.add_patch(rect)
    ax.annotate(f"{speed:.2f}", (rx, ry), fontsize=7, ha="center", color="darkred", zorder=12)
part_mask = np.any(partners != 0, axis=1)
if part_mask.any():
    ax.scatter(
        partners[part_mask, 0],
        partners[part_mask, 1],
        s=40,
        color="orange",
        edgecolors="black",
        label=f"partners ({n_visible})",
        zorder=8,
    )

# Target waypoints
for wp in range(target.shape[0]):
    if np.allclose(target[wp], 0):
        continue
    marker = "*" if wp == 0 else "o"
    s = 200 if wp == 0 else 80
    color = "red" if wp == 0 else "salmon"
    ax.scatter(
        target[wp, 0],
        target[wp, 1],
        color=color,
        marker=marker,
        s=s,
        zorder=15,
        label=f"target wp[{wp}]" if wp < 3 else None,
    )

# Traffic controls
for i in range(traffic_controls.shape[0]):
    if np.allclose(traffic_controls[i], 0):
        continue
    x1, y1, x2, y2, _, control_type, state = traffic_controls[i]
    if int(control_type) == binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT:
        state_colors = {
            binding.TRAFFIC_CONTROL_STATE_UNKNOWN: "gray",
            binding.TRAFFIC_CONTROL_STATE_RED: "red",
            binding.TRAFFIC_CONTROL_STATE_YELLOW: "yellow",
            binding.TRAFFIC_CONTROL_STATE_GREEN: "green",
            binding.TRAFFIC_CONTROL_STATE_OFF: "gray",
        }
        ax.plot([x1, x2], [y1, y2], color=state_colors.get(int(state), "gray"), linewidth=3, zorder=15)
    else:
        accent = "red" if int(control_type) == binding.TRAFFIC_CONTROL_TYPE_STOP_SIGN else "gold"
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=4, zorder=14)
        ax.plot([x1, x2], [y1, y2], color=accent, linewidth=2.5, linestyle="--", zorder=15)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal")
ax.set_xlabel("X (ego frame, normalized)")
ax.set_ylabel("Y (ego frame, normalized)")
ax.set_title(f"All observed entities | agent={TRACKED_AGENT}, t={sample_t}")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Ego + conditioning distributions across all agents

# %%
# Ego feature distributions across all agents, pooled over full rollout
ego_dim = binding.EGO_FEATURES
all_ego = buf_stoch["obs"][:, :, :ego_dim].reshape(-1, ego_dim)  # (H*N, ego_dim)

ego_labels = EGO_LABELS

fig, axes = plt.subplots(2, len(ego_labels), figsize=(3.5 * len(ego_labels), 7))

# Row 0: histograms
for i, label in enumerate(ego_labels):
    vals = all_ego[:, i]
    print(f"{label}: mean={vals}")
    axes[0, i].hist(vals, bins=60, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, i].set_title(label, fontsize=10)
    axes[0, i].set_xlabel("")
    axes[0, i].tick_params(labelsize=7)
    axes[0, i].axvline(vals.mean(), color="red", ls="--", lw=1)

# Row 1: boxplots per-agent (distribution across timesteps for each agent)
ego_per_agent = buf_stoch["obs"][:, :, :ego_dim]  # (H, N, ego_dim)
for i, label in enumerate(ego_labels):
    data = [ego_per_agent[:, a, i] for a in range(N)]
    bp = axes[1, i].boxplot(
        data,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.5),
        medianprops=dict(color="red"),
    )
    axes[1, i].set_xlabel("Agent")
    axes[1, i].tick_params(labelsize=7)
    axes[1, i].set_title(f"{label} per agent", fontsize=9)

fig.suptitle("Ego features: full rollout distributions", fontsize=13)
plt.tight_layout()
plt.show()

# Conditioning distributions across all agents (if enabled)
if rew_cond:
    cond_start = ego_dim
    cond_end = cond_start + binding.NUM_REWARD_COEFS
    all_cond = buf_stoch["obs"][:, :, cond_start:cond_end].reshape(-1, binding.NUM_REWARD_COEFS)

    cond_labels = COEF_NAMES

    fig, ax = plt.subplots(figsize=(14, 5))
    parts = ax.violinplot(
        [all_cond[:, i] for i in range(binding.NUM_REWARD_COEFS)],
        positions=range(binding.NUM_REWARD_COEFS),
        showmeans=True,
        showmedians=True,
    )
    ax.set_xticks(range(binding.NUM_REWARD_COEFS))
    ax.set_xticklabels(cond_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Normalized value")
    ax.set_title("Conditioning coef distributions (all agents, full rollout)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Partner per-feature distributions

# %%
# Partner per-feature distributions (pooled over all agents + timesteps, visible only)
partner_labels = [
    "rel_x",
    "rel_y",
    "rel_z",
    "length",
    "width",
    "heading_cos",
    "heading_sin",
    "sim_speed_signed",
    "seconds_stopped",
]
obs_slots_partners_n = env.obs_slots_partners_n
pf = env.partner_features

# Compute slices
_ego_d = binding.EGO_FEATURES
_cond_d = binding.NUM_REWARD_COEFS if rew_cond else 0
_tgt_f = binding.STATIC_TARGET_FEATURES if tgt_type == "static" else binding.DYNAMIC_TARGET_FEATURES
_tgt_d = n_tgt_wp * _tgt_f
_p_start = _ego_d + _cond_d + _tgt_d
_p_end = _p_start + obs_slots_partners_n * pf

all_partners = buf_stoch["obs"][:, :, _p_start:_p_end].reshape(
    -1, obs_slots_partners_n, pf
)  # (H*N, obs_slots_partners_n, pf)
# Mask: partner is visible if any feature != 0
visible_mask = np.any(all_partners != 0, axis=2)  # (H*N, 16)
visible_partners = all_partners[visible_mask]  # (K, pf) — all visible partner observations

print(
    f"Total partner obs: {all_partners.shape[0] * obs_slots_partners_n}, visible: {len(visible_partners)} "
    f"({100 * len(visible_partners) / (all_partners.shape[0] * obs_slots_partners_n):.1f}%)"
)

fig, axes = plt.subplots(3, 4, figsize=(21, 11))
axes = axes.flatten()

for i, label in enumerate(partner_labels):
    vals = visible_partners[:, i]
    axes[i].hist(vals, bins=80, edgecolor="black", alpha=0.7, color="darkorange")
    axes[i].set_title(f"{label} (n={len(vals)})", fontsize=10)
    axes[i].axvline(vals.mean(), color="red", ls="--", lw=1, label=f"mean={vals.mean():.3f}")
    axes[i].legend(fontsize=7)
    axes[i].tick_params(labelsize=7)

# rel_x vs rel_y scatter in last panel
pos_ax = axes[len(partner_labels)]
pos_ax.scatter(visible_partners[:, 0], visible_partners[:, 1], s=1, alpha=0.15, color="darkorange")
pos_ax.set_xlabel("rel_x")
pos_ax.set_ylabel("rel_y")
pos_ax.set_title("Partner positions (ego frame)")
pos_ax.set_aspect("equal")
pos_ax.grid(True, alpha=0.3)

for ax in axes[len(partner_labels) + 1 :]:
    ax.axis("off")

fig.suptitle("Partner features: all visible, full rollout", fontsize=13)
plt.tight_layout()
plt.show()

# Partner count distribution across (timestep, agent)
partner_counts = visible_mask.sum(axis=1)  # (H*N,)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(partner_counts, bins=range(obs_slots_partners_n + 2), edgecolor="black", alpha=0.7, color="darkorange")
axes[0].set_xlabel("Visible partners")
axes[0].set_ylabel("Count")
axes[0].set_title("Partner count distribution (per agent per step)")

# Partner distance distribution
dists = np.sqrt(visible_partners[:, 0] ** 2 + visible_partners[:, 1] ** 2)
axes[1].hist(dists, bins=80, edgecolor="black", alpha=0.7, color="coral")
axes[1].set_xlabel("Distance (normalized)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Partner distance distribution (mean={dists.mean():.3f})")
axes[1].axvline(dists.mean(), color="red", ls="--", lw=1)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Road (lanes + boundaries) and target distributions

# %%
# Road per-feature distributions (lanes + boundaries)
road_labels = ["rel_x", "rel_y", "rel_z", "seg_length", "seg_width", "dir_cos", "dir_sin"]
rf = env.road_features
max_lanes = env.obs_slots_lane_kept
max_bounds = env.obs_slots_boundary_kept

_l_start = _p_end
_l_end = _l_start + max_lanes * rf
_b_start = _l_end
_b_end = _b_start + max_bounds * rf

all_lanes = buf_stoch["obs"][:, :, _l_start:_l_end].reshape(-1, max_lanes, rf)
all_bounds = buf_stoch["obs"][:, :, _b_start:_b_end].reshape(-1, max_bounds, rf)

vis_lanes = all_lanes[np.any(all_lanes != 0, axis=2)]
vis_bounds = all_bounds[np.any(all_bounds != 0, axis=2)]

print(
    f"Lanes: {len(vis_lanes)} visible / {all_lanes.shape[0] * max_lanes} total "
    f"({100 * len(vis_lanes) / (all_lanes.shape[0] * max_lanes):.1f}%)"
)
print(
    f"Boundaries: {len(vis_bounds)} visible / {all_bounds.shape[0] * max_bounds} total "
    f"({100 * len(vis_bounds) / (all_bounds.shape[0] * max_bounds):.1f}%)"
)

fig, axes = plt.subplots(2, 7, figsize=(28, 8))
for i, label in enumerate(road_labels):
    # Lanes
    axes[0, i].hist(vis_lanes[:, i], bins=80, edgecolor="black", alpha=0.7, color="silver")
    axes[0, i].set_title(f"lane {label}", fontsize=9)
    axes[0, i].axvline(vis_lanes[:, i].mean(), color="red", ls="--", lw=1)
    axes[0, i].tick_params(labelsize=7)
    # Boundaries
    axes[1, i].hist(vis_bounds[:, i], bins=80, edgecolor="black", alpha=0.7, color="dimgray")
    axes[1, i].set_title(f"boundary {label}", fontsize=9)
    axes[1, i].axvline(vis_bounds[:, i].mean(), color="red", ls="--", lw=1)
    axes[1, i].tick_params(labelsize=7)

fig.suptitle("Road features: all visible, full rollout (top=lanes, bottom=boundaries)", fontsize=13)
plt.tight_layout()
plt.show()

# Spatial scatter: lane vs boundary positions (pooled)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(vis_lanes[:, 0], vis_lanes[:, 1], s=0.5, alpha=0.05, color="gray")
axes[0].set_xlabel("rel_x")
axes[0].set_ylabel("rel_y")
axes[0].set_title(f"Lane segment positions (n={len(vis_lanes)})")
axes[0].set_aspect("equal")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(vis_bounds[:, 0], vis_bounds[:, 1], s=0.5, alpha=0.05, color="black")
axes[1].set_xlabel("rel_x")
axes[1].set_ylabel("rel_y")
axes[1].set_title(f"Boundary segment positions (n={len(vis_bounds)})")
axes[1].set_aspect("equal")
axes[1].grid(True, alpha=0.3)

# Lane + boundary segment length comparison
axes[2].hist(vis_lanes[:, 2], bins=80, alpha=0.6, color="silver", edgecolor="black", label="lanes")
axes[2].hist(vis_bounds[:, 2], bins=80, alpha=0.6, color="dimgray", edgecolor="black", label="boundaries")
axes[2].set_xlabel("Segment length (normalized)")
axes[2].set_ylabel("Count")
axes[2].set_title("Segment length distribution")
axes[2].legend()

plt.tight_layout()
plt.show()

# Target distributions across all agents, full rollout
_tgt_start = _ego_d + _cond_d
_tgt_end = _tgt_start + _tgt_d
all_target = buf_stoch["obs"][:, :, _tgt_start:_tgt_end].reshape(-1, n_tgt_wp, _tgt_f)

if tgt_type == "static":
    tgt_flabels = ["rel_x", "rel_y", "rel_z"]
else:
    tgt_flabels = ["rel_x", "rel_y", "rel_z", "heading_cos", "heading_sin"]

fig, axes = plt.subplots(1, n_tgt_wp + 1, figsize=(5 * (n_tgt_wp + 1), 4))

for wp in range(n_tgt_wp):
    wp_data = all_target[:, wp, :]
    active = np.any(wp_data != 0, axis=1)
    wp_active = wp_data[active]
    dist = np.sqrt(wp_active[:, 0] ** 2 + wp_active[:, 1] ** 2) if len(wp_active) > 0 else np.array([])
    axes[wp].hist(dist, bins=60, edgecolor="black", alpha=0.7, color=["red", "salmon", "lightsalmon"][wp % 3])
    axes[wp].set_title(f"wp[{wp}] distance (n={len(wp_active)}/{len(wp_data)})", fontsize=10)
    axes[wp].set_xlabel("Distance (normalized)")

# All waypoints x-y scatter
for wp in range(n_tgt_wp):
    wp_data = all_target[:, wp, :]
    active = np.any(wp_data != 0, axis=1)
    wp_active = wp_data[active]
    if len(wp_active) > 0:
        axes[n_tgt_wp].scatter(wp_active[:, 0], wp_active[:, 1], s=1, alpha=0.1, label=f"wp[{wp}]")
axes[n_tgt_wp].set_xlabel("rel_x")
axes[n_tgt_wp].set_ylabel("rel_y")
axes[n_tgt_wp].set_title("Target positions (ego frame)")
axes[n_tgt_wp].set_aspect("equal")
axes[n_tgt_wp].legend(fontsize=8)
axes[n_tgt_wp].grid(True, alpha=0.3)

fig.suptitle("Target waypoint distributions (all agents, full rollout)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation sparsity and layer occupancy heatmaps

# %%
# Sparsity heatmap: fraction of nonzero per layer, per agent, over time
layer_names = ["partners", "lanes", "boundaries"]
layer_slices = [
    (_p_start, _p_end, env.obs_slots_partners_n, env.partner_features),
    (_l_start, _l_end, env.obs_slots_lane_kept, env.road_features),
    (_b_start, _b_end, env.obs_slots_boundary_kept, env.road_features),
]

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, name, (s, e, n_obj, n_feat) in zip(axes, layer_names, layer_slices):
    # (H, N) -> fraction of visible objects per (timestep, agent)
    raw = buf_stoch["obs"][:, :, s:e].reshape(HORIZON, env.num_agents, n_obj, n_feat)
    occupancy = np.any(raw != 0, axis=3).sum(axis=2) / n_obj  # (H, N)
    im = ax.imshow(occupancy.T, aspect="auto", cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Agent")
    ax.set_title(f"{name} occupancy (frac visible)")
    plt.colorbar(im, ax=ax)

plt.suptitle("Per-layer occupancy heatmaps (fraction of max slots filled)", fontsize=13)
plt.tight_layout()
plt.show()

# Per-layer mean occupancy over time
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Mean across agents
for name, (s, e, n_obj, n_feat) in zip(layer_names, layer_slices):
    raw = buf_stoch["obs"][:, :, s:e].reshape(HORIZON, env.num_agents, n_obj, n_feat)
    occ_mean = np.any(raw != 0, axis=3).sum(axis=2).mean(axis=1)  # (H,)
    axes[0].plot(occ_mean, label=name, alpha=0.8)
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Mean visible count")
axes[0].set_title("Mean occupancy over time (across agents)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Mean across timesteps (per agent)
for name, (s, e, n_obj, n_feat) in zip(layer_names, layer_slices):
    raw = buf_stoch["obs"][:, :, s:e].reshape(HORIZON, env.num_agents, n_obj, n_feat)
    occ_per_agent = np.any(raw != 0, axis=3).sum(axis=2).mean(axis=0)  # (N,)
    axes[1].bar(range(N), occ_per_agent, alpha=0.5, label=name)
axes[1].set_xlabel("Agent")
axes[1].set_ylabel("Mean visible count")
axes[1].set_title("Mean occupancy per agent (across timesteps)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Full obs sparsity: fraction of zero features per obs dimension, pooled
all_flat = buf_stoch["obs"].reshape(-1, obs_dim)  # (H*N, obs_dim)
zero_frac = (all_flat == 0).mean(axis=0)  # (obs_dim,)
fig, ax = plt.subplots(figsize=(18, 3))
ax.bar(range(obs_dim), zero_frac, width=1.0, color="steelblue", alpha=0.7)
# Annotate layer boundaries
prev_e = 0
for name, (s, e) in slices.items():
    ax.axvline(s, color="red", ls="--", lw=0.5, alpha=0.7)
    mid = (s + e) / 2
    ax.text(mid, 1.02, name, ha="center", va="bottom", fontsize=7, rotation=0, color="red")
    prev_e = e
ax.set_xlim(0, obs_dim)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Obs dimension index")
ax.set_ylabel("Fraction zero")
ax.set_title("Per-dimension sparsity (fraction zero across full rollout)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Policy outputs over time

# %%
# Compute action probs over time for tracked agent (stochastic rollout)
n_actions = env.single_action_space.nvec[0] if not is_continuous else 1
action_probs_time = np.zeros((HORIZON, n_actions))
for t in range(HORIZON):
    obs_t = torch.FloatTensor(buf_stoch["obs"][t : t + 1, TRACKED_AGENT : TRACKED_AGENT + 1][0]).to(device)
    with torch.no_grad():
        logits_list, _ = policy(obs_t)
    logits = logits_list[0] if isinstance(logits_list, (list, tuple)) else logits_list
    action_probs_time[t] = F.softmax(logits, dim=-1).cpu().numpy().flatten()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Action distribution heatmap
im = axes[0, 0].imshow(action_probs_time.T, aspect="auto", cmap="viridis", interpolation="nearest")
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Action ID")
axes[0, 0].set_title(f"Action prob heatmap | agent={TRACKED_AGENT}")
plt.colorbar(im, ax=axes[0, 0])

# Entropy over time
axes[0, 1].plot(buf_stoch["entropy"][:, TRACKED_AGENT], label="stochastic", alpha=0.8)
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Entropy")
axes[0, 1].set_title("Entropy over time")
axes[0, 1].grid(True, alpha=0.3)

# Value over time
axes[1, 0].plot(buf_stoch["values"][:, TRACKED_AGENT], label="stochastic", alpha=0.8)
axes[1, 0].plot(buf_det["values"][:, TRACKED_AGENT], label="deterministic", alpha=0.8)
axes[1, 0].set_xlabel("Step")
axes[1, 0].set_ylabel("Value")
axes[1, 0].set_title("Value predictions over time")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Actions over time: deterministic vs stochastic
axes[1, 1].step(range(HORIZON), buf_stoch["actions"][:, TRACKED_AGENT], label="stochastic", alpha=0.7)
axes[1, 1].step(range(HORIZON), buf_det["actions"][:, TRACKED_AGENT], label="deterministic", alpha=0.7)
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_ylabel("Action")
axes[1, 1].set_title("Selected action over time")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Rewards and returns

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Per-step mean reward
axes[0, 0].plot(buf_stoch["rewards"].mean(axis=1), label="stochastic", alpha=0.8)
axes[0, 0].plot(buf_det["rewards"].mean(axis=1), label="deterministic", alpha=0.8)
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Mean reward")
axes[0, 0].set_title("Mean reward per step")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Reward heatmap (stochastic)
im = axes[0, 1].imshow(buf_stoch["rewards"].T, aspect="auto", cmap="RdYlGn", interpolation="nearest")
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Agent")
axes[0, 1].set_title("Reward heatmap (stochastic)")
plt.colorbar(im, ax=axes[0, 1])

# Cumulative return per agent
cum_ret_stoch = buf_stoch["rewards"].sum(axis=0)
cum_ret_det = buf_det["rewards"].sum(axis=0)
axes[0, 2].hist(cum_ret_stoch, bins=30, alpha=0.6, label="stochastic", edgecolor="black")
axes[0, 2].hist(cum_ret_det, bins=30, alpha=0.6, label="deterministic", edgecolor="black")
axes[0, 2].set_xlabel("Cumulative return")
axes[0, 2].set_ylabel("Count")
axes[0, 2].set_title("Return distribution across agents")
axes[0, 2].legend()

# Reward distribution histogram
axes[1, 0].hist(buf_stoch["rewards"].flatten(), bins=50, alpha=0.7, edgecolor="black")
axes[1, 0].set_xlabel("Reward")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Per-step reward distribution (stochastic)")
axes[1, 0].set_yscale("log")

# Terminal/truncation timeline
axes[1, 1].plot(buf_stoch["terminals"].sum(axis=1), label="terminals", alpha=0.8)
axes[1, 1].plot(buf_stoch["truncations"].sum(axis=1), label="truncations", alpha=0.8)
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("Terminals/Truncations per step (stochastic)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Tracked agent reward
axes[1, 2].plot(buf_stoch["rewards"][:, TRACKED_AGENT], label="stochastic", alpha=0.8)
axes[1, 2].plot(buf_det["rewards"][:, TRACKED_AGENT], label="deterministic", alpha=0.8)
axes[1, 2].set_xlabel("Step")
axes[1, 2].set_ylabel("Reward")
axes[1, 2].set_title(f"Reward over time | agent={TRACKED_AGENT}")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Episode metrics

# %%
# Collect episode-level metrics from the C logging
log_stoch = binding.vec_log(env.c_envs, N)

# eval_mode=1 returns a list of per-env dicts; aggregate by averaging
if isinstance(log_stoch, list) and log_stoch:
    all_keys = set(k for d in log_stoch for k in d if isinstance(d[k], (int, float)))
    log_stoch = {k: np.mean([d[k] for d in log_stoch if k in d]) for k in all_keys}

if log_stoch:
    print("Episode metrics (after stochastic rollout):")
    for k, v in sorted(log_stoch.items()):
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")

    # Bar chart of key metrics
    keys = ["score", "collision_rate", "offroad_rate", "completion_rate", "dnf_rate"]
    vals = [log_stoch.get(k, 0) for k in keys]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(keys, vals, edgecolor="black", alpha=0.7, color=["green", "red", "orange", "blue", "gray"])
    ax.set_ylabel("Rate")
    ax.set_title("Episode Metrics")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.show()
else:
    print("No episode metrics available yet (not enough episodes completed)")

# %% [markdown]
# ## Value predictions vs actual returns

# %%
gamma = config["train"].get("gamma", 0.98)
lam = config["train"].get("gae_lambda", 0.95)


def compute_gae(rewards, values, terminals, truncations, gamma, lam):
    H, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(N)
    for t in reversed(range(H - 1)):
        done = np.maximum(terminals[t + 1], truncations[t + 1])
        next_non_terminal = 1.0 - done
        delta = rewards[t + 1] + gamma * values[t + 1] * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * last_gae * next_non_terminal
        advantages[t] = last_gae
    return advantages


adv_stoch = compute_gae(
    buf_stoch["rewards"], buf_stoch["values"], buf_stoch["terminals"], buf_stoch["truncations"], gamma, lam
)
returns_stoch = adv_stoch + buf_stoch["values"]

pred_v = buf_stoch["values"].flatten()
actual_r = returns_stoch.flatten()

var_actual = np.var(actual_r)
explained_var = 1 - np.var(actual_r - pred_v) / (var_actual + 1e-8) if var_actual > 1e-8 else 0.0

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter: predicted vs actual
axes[0].scatter(actual_r, pred_v, alpha=0.2, s=5)
lims = [min(actual_r.min(), pred_v.min()), max(actual_r.max(), pred_v.max())]
axes[0].plot(lims, lims, "r--", label="perfect")
axes[0].set_xlabel("Actual return")
axes[0].set_ylabel("Predicted value")
axes[0].set_title(f"Value accuracy (EV: {explained_var:.4f})")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Value error over time
value_error = np.abs(returns_stoch - buf_stoch["values"]).mean(axis=1)
axes[1].plot(value_error)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Mean |error|")
axes[1].set_title("Value prediction error over time")
axes[1].grid(True, alpha=0.3)

# Advantage distribution
axes[2].hist(adv_stoch.flatten(), bins=50, edgecolor="black", alpha=0.7)
axes[2].set_xlabel("Advantage")
axes[2].set_ylabel("Count")
axes[2].set_title(f"Advantage distribution (std={adv_stoch.std():.4f})")

plt.tight_layout()
plt.show()

print(f"Explained variance: {explained_var:.4f}")
print(f"Value MSE: {np.mean((actual_r - pred_v) ** 2):.6f}")

# %% [markdown]
# ## Agent trajectories

# %%
N_TRAJ = min(16, N)  # number of agents to plot

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for i in range(N_TRAJ):
    color = plt.cm.tab20(i % 20)
    # Stochastic
    axes[0].plot(buf_stoch["positions_x"][:, i], buf_stoch["positions_y"][:, i], alpha=0.6, color=color, linewidth=1)
    axes[0].scatter(
        buf_stoch["positions_x"][0, i], buf_stoch["positions_y"][0, i], color=color, s=30, marker="o", zorder=5
    )  # start
    # Deterministic
    axes[1].plot(buf_det["positions_x"][:, i], buf_det["positions_y"][:, i], alpha=0.6, color=color, linewidth=1)
    axes[1].scatter(buf_det["positions_x"][0, i], buf_det["positions_y"][0, i], color=color, s=30, marker="o", zorder=5)

axes[0].set_title(f"Stochastic trajectories (N={N_TRAJ})")
axes[1].set_title(f"Deterministic trajectories (N={N_TRAJ})")
for ax in axes:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ADE vs ground truth if scenario_length is set
if config["env"].get("scenario_length"):
    try:
        gt = env.get_ground_truth_trajectories()
        # gt['x'] shape: (N, 1, T), positions shape: (T, N)
        gt_x = gt["x"][:, 0, :].T  # (T, N)
        gt_y = gt["y"][:, 0, :].T
        gt_valid = gt["valid"][:, 0, :].T
        T_gt = gt_x.shape[0]
        T_use = min(T_gt, HORIZON)

        disp = np.sqrt(
            (buf_stoch["positions_x"][:T_use] - gt_x[:T_use]) ** 2
            + (buf_stoch["positions_y"][:T_use] - gt_y[:T_use]) ** 2
        )
        valid_mask = gt_valid[:T_use] > 0
        if valid_mask.sum() > 0:
            ade = disp[valid_mask].mean()
            print(f"ADE (stochastic vs ground truth): {ade:.3f}m")
            ade_per_agent = np.array(
                [disp[:, i][valid_mask[:, i]].mean() for i in range(N) if valid_mask[:, i].sum() > 0]
            )
            plt.figure(figsize=(8, 3))
            plt.hist(ade_per_agent, bins=30, edgecolor="black", alpha=0.7)
            plt.xlabel("ADE (m)")
            plt.ylabel("Count")
            plt.title(f"Per-agent ADE distribution (mean={ade:.3f}m)")
            plt.tight_layout()
            plt.show()
        else:
            print("No valid ground truth timesteps to compute ADE")
    except Exception as e:
        print(f"Could not compute ADE: {e}")

# %% [markdown]
# ## Encoder analysis — what the policy encodes
#
# Each obs layer has its own encoder projecting raw features → embedding width:
# - **ego** and **conditioning** (reward coefs + target): single vector, no pooling.
# - **partners / lanes / boundaries / traffic**: per-slot encoder, padded slots masked to `-inf`, then **max-pooled** across slots → one embedding. Fully-padded layers are zeroed.
#
# The max-pool means each embedding dim is "won" by exactly one slot (object). Below we inspect:
# 1. Encoder inventory (in/out dims, params).
# 2. **What survives the max-pool**: which slot wins per dim, per-dim winner entropy (slot-specialized vs. spread), and where the dominant objects sit in ego frame.
# 3. **Embedding space**: per-encoder contribution (L2 norm), active/dead dims, silence rate.

# %%
# ── Setup: capture per-encoder embeddings + reconstruct the max-pool ──
bb = policy.actor_backbone
ego_dim = policy.ego_dim
PAD = -1.0  # PADDED_OBSERVATION_VALUE

# Flat batch of observations from the stochastic rollout
obs_flat = buf_stoch["obs"].reshape(-1, obs_dim)
rng = np.random.default_rng(0)
sel = rng.choice(obs_flat.shape[0], size=min(4096, obs_flat.shape[0]), replace=False)
obs_batch = torch.FloatTensor(obs_flat[sel]).to(device)
B = obs_batch.shape[0]

# Encoder inventory: (name, module, raw_in_features, n_slots, is_set)
enc_inventory = [("ego", bb.ego_encoder, ego_dim, 1, False)]
if bb.obs_slots_lane_kept > 0:
    enc_inventory.append(("lane", bb.lane_encoder, bb.road_features_count, bb.obs_slots_lane_kept, True))
if bb.obs_slots_boundary_kept > 0:
    enc_inventory.append(("boundary", bb.boundary_encoder, bb.road_features_count, bb.obs_slots_boundary_kept, True))
if bb.obs_slots_partners_n > 0:
    enc_inventory.append(("partner", bb.partner_encoder, bb.partner_features_count, bb.obs_slots_partners_n, True))
if bb.obs_slots_traffic_controls_n > 0:
    enc_inventory.append(
        (
            "traffic",
            bb.traffic_control_encoder,
            bb.traffic_control_features_after_onehot,
            bb.obs_slots_traffic_controls_n,
            True,
        )
    )
if bb.context_dim > 0:
    enc_inventory.append(("context", bb.context_encoder, bb.context_dim, 1, False))

enc_names = [n for n, *_ in enc_inventory]
set_encs = [n for n, _, _, _, is_set in enc_inventory if is_set]

print(f"{'encoder':>13s} | {'raw_in':>6s} | {'emb_out':>7s} | {'slots':>5s} | {'pooled':>6s} | {'params':>9s}")
print("-" * 66)
for name, mod, rin, nslots, is_set in enc_inventory:
    nparam = sum(p.numel() for p in mod.parameters())
    print(
        f"{name:>13s} | {rin:>6d} | {mod[-1].out_features:>7d} | {nslots:>5d} | {('max' if is_set else '-'):>6s} | {nparam:>9,d}"
    )
print(
    f"\nBackbone input = {sum(mod[-1].out_features for _, mod, _, _, _ in enc_inventory)} -> backbone -> {bb.out_dim}"
)

# Capture pre-pool encoder outputs via forward hooks
captured = {}


def _hook(name):
    def fn(m, i, o):
        captured[name] = o.detach()

    return fn


handles = [mod.register_forward_hook(_hook(name)) for name, mod, *_ in enc_inventory]
policy.eval()
with torch.no_grad():
    policy(obs_batch)
for h in handles:
    h.remove()

# Reconstruct slot slices (same order as DriveBackbone.forward) + pad masks
partner_dim = bb.obs_slots_partners_n * bb.partner_features_count
lane_dim = bb.obs_slots_lane_kept * bb.road_features_count
boundary_dim = bb.obs_slots_boundary_kept * bb.road_features_count
traffic_dim = bb.obs_slots_traffic_controls_n * bb.traffic_control_features_count
_s = ego_dim + bb.context_dim
sl = {}
sl["partner"] = (_s, _s + partner_dim, bb.obs_slots_partners_n, bb.partner_features_count)
_s += partner_dim
sl["lane"] = (_s, _s + lane_dim, bb.obs_slots_lane_kept, bb.road_features_count)
_s += lane_dim
sl["boundary"] = (_s, _s + boundary_dim, bb.obs_slots_boundary_kept, bb.road_features_count)
_s += boundary_dim
sl["traffic"] = (_s, _s + traffic_dim, bb.obs_slots_traffic_controls_n, bb.traffic_control_features_count)
_s += traffic_dim

raw, pad, pooled, winners, valid_sample = {}, {}, {}, {}, {}
for name in set_encs:
    s, e, ns, nf = sl[name]
    obj = obs_batch[:, s:e].view(B, ns, nf)
    raw[name] = obj
    if name == "traffic":
        cont = obj[:, :, : bb.traffic_control_continuous_features]
        typ = obj[:, :, bb.traffic_control_continuous_features]
        st = obj[:, :, bb.traffic_control_continuous_features + 1]
        pad[name] = (
            (cont == PAD).all(dim=2)
            & (typ == binding.TRAFFIC_CONTROL_TYPE_NONE)
            & (st == binding.TRAFFIC_CONTROL_STATE_UNKNOWN)
        )
    else:
        pad[name] = (obj == PAD).all(dim=2)
    masked = captured[name].masked_fill(pad[name].unsqueeze(2), -torch.inf)
    vm = (~pad[name]).any(dim=1)
    valid_sample[name] = vm
    winners[name] = masked.max(dim=1).indices  # (B, embedding dim): winning slot per dim
    pooled[name] = torch.where(vm.unsqueeze(1), masked.max(dim=1).values, torch.zeros_like(masked.max(dim=1).values))

for name in ("ego", "context"):
    if name in enc_names:
        pooled[name] = captured[name]

print("\nCaptured embeddings for:", enc_names)

# %%
# ── What survives the max-pool: winning slots, specialization, spatial ──
n = len(set_encs)
fig, axes = plt.subplots(n, 3, figsize=(18, 4.2 * n))
if n == 1:
    axes = axes[None, :]

print(f"{'encoder':>9s} | {'valid%':>6s} | {'mean active slots/dim':>21s} | {'%slot-specialized dims':>22s}")
print("-" * 70)
for r, name in enumerate(set_encs):
    s, e, ns, nf = sl[name]
    vm = valid_sample[name]
    w = winners[name][vm]  # (Bv, D)
    D = w.shape[1]

    # (1) which slot wins, pooled over all dims+samples
    slot_counts = torch.bincount(w.reshape(-1), minlength=ns).float().cpu().numpy()
    slot_counts = slot_counts / max(slot_counts.sum(), 1)
    axes[r, 0].bar(range(ns), slot_counts, color="teal", alpha=0.85, edgecolor="black")
    axes[r, 0].set_title(f"{name}: max-pool winner by slot")
    axes[r, 0].set_xlabel("slot index (0 = first/closest)")
    axes[r, 0].set_ylabel("frac of dims won")

    # (2) per-dim winner entropy: slot-specialized (0) vs spread across slots (1)
    onehot = F.one_hot(w, num_classes=ns).float()  # (Bv, D, ns)
    p = onehot.mean(dim=0)  # (D, ns) winner distribution per dim
    ent = (-(p * (p + 1e-9).log()).sum(dim=1) / np.log(ns)).cpu().numpy()
    axes[r, 1].hist(ent, bins=30, color="indianred", alpha=0.85, edgecolor="black")
    axes[r, 1].set_title(f"{name}: per-dim winner entropy")
    axes[r, 1].set_xlabel("0 = slot-specialized   →   1 = spread")
    axes[r, 1].set_xlim(0, 1)

    # (3) ego-frame position of the dominant object (mode winning slot per sample)
    dom = torch.mode(w, dim=1).values  # (Bv,)
    rel = raw[name][vm]
    dom_xy = rel[torch.arange(rel.shape[0]), dom][:, :2].cpu().numpy()
    axes[r, 2].scatter(dom_xy[:, 0], dom_xy[:, 1], s=3, alpha=0.15, color="navy")
    axes[r, 2].scatter(0, 0, marker="*", s=200, color="red", zorder=5, label="ego")
    axes[r, 2].set_title(f"{name}: dominant object position (ego frame)")
    axes[r, 2].set_xlabel("rel_x")
    axes[r, 2].set_ylabel("rel_y")
    axes[r, 2].set_aspect("equal")
    axes[r, 2].legend(fontsize=8)

    active_per_dim = np.exp(ent * np.log(ns)).mean()
    print(
        f"{name:>9s} | {100 * vm.float().mean().item():>5.1f}% | {active_per_dim:>21.2f} | {100 * (ent < 0.2).mean():>21.1f}%"
    )

plt.tight_layout()
plt.show()


# ── H1/H2/H3 check: boundary max-pool winner distance vs slot index ──
if "boundary" in set_encs:
    vm = valid_sample["boundary"]
    w = winners["boundary"][vm]  # (Bv, D) winning slot per dim
    rb = raw["boundary"][vm]  # (Bv, ns, nf) raw segments
    nsb = rb.shape[1]
    reldist = torch.hypot(rb[:, :, 0], rb[:, :, 1])  # (Bv, ns) normalized ego-frame dist
    valid_seg = ~pad["boundary"][vm]  # (Bv, ns) slots holding a real segment

    win_reldist = torch.gather(reldist, 1, w)  # (Bv, D) dist of each winning segment
    slot_flat = w.reshape(-1)
    wdist_flat = win_reldist.reshape(-1)
    total_wins = slot_flat.numel()

    print("\n=== Boundary winner distance vs slot index (H1/H2/H3) ===")
    print(f"{'slot':>4s} | {'#wins':>8s} | {'win%':>6s} | {'rel_dist winners':>16s} | {'rel_dist occupied':>17s}")
    print("-" * 64)
    for s in range(nsb):
        wm = slot_flat == s
        nwin = int(wm.sum())
        wmean = wdist_flat[wm].mean().item() if nwin > 0 else float("nan")
        occ = valid_seg[:, s]
        omean = reldist[occ, s].mean().item() if int(occ.sum()) > 0 else float("nan")
        print(f"{s:>4d} | {nwin:>8d} | {100 * nwin / total_wins:>5.1f}% | {wmean:>16.4f} | {omean:>17.4f}")

    win_mean = wdist_flat.mean().item()
    seg_mean = reldist[valid_seg].mean().item()
    print(f"\nMean rel_dist of WINNING segments : {win_mean:.4f}")
    print(f"Mean rel_dist of ALL valid segments: {seg_mean:.4f}")
    verdict = "FARTHER than avg (H3 supported)" if win_mean > seg_mean else "nearer than avg"
    print(f"-> winners are {verdict} by {win_mean - seg_mean:+.4f} (normalized units)")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    occ_means = [
        reldist[valid_seg[:, s], s].mean().item() if int(valid_seg[:, s].sum()) > 0 else np.nan for s in range(nsb)
    ]
    win_means = [
        wdist_flat[slot_flat == s].mean().item() if int((slot_flat == s).sum()) > 0 else np.nan for s in range(nsb)
    ]
    ax[0].plot(range(nsb), occ_means, "o-", label="occupied (any segment in slot)")
    ax[0].plot(range(nsb), win_means, "s-", label="winners only")
    ax[0].axhline(seg_mean, color="gray", ls="--", label="global valid mean")
    ax[0].set_xlabel("slot index")
    ax[0].set_ylabel("mean rel_dist (normalized)")
    ax[0].set_title("Boundary rel_dist vs slot index\n(flat = not distance-sorted -> H1/H2)")
    ax[0].legend(fontsize=8)
    ax[1].hist(reldist[valid_seg].cpu().numpy(), bins=50, alpha=0.6, density=True, label="all valid segs", color="gray")
    ax[1].hist(wdist_flat.cpu().numpy(), bins=50, alpha=0.6, density=True, label="winners", color="crimson")
    ax[1].axvline(seg_mean, color="gray", ls="--")
    ax[1].axvline(win_mean, color="crimson", ls="--")
    ax[1].set_xlabel("rel_dist (normalized)")
    ax[1].set_title("Winner vs all-segment distance (H3)")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# %%
# ── Embedding space: per-encoder contribution, active/dead dims, silence ──
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# (1) Mean L2 norm of each pooled embedding = relative weight in the concat fed to backbone
norms = [pooled[n].norm(dim=1).mean().item() for n in enc_names]
axes[0].bar(enc_names, norms, color="slateblue", edgecolor="black")
axes[0].set_title("Mean L2 norm of pooled embedding\n(relative contribution to backbone input)")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(True, axis="y", alpha=0.3)

# (2) Mean |activation| per embedding dim, per encoder
M = np.stack([pooled[n].abs().mean(0).cpu().numpy() for n in enc_names])
im = axes[1].imshow(M, aspect="auto", cmap="magma")
axes[1].set_yticks(range(len(enc_names)))
axes[1].set_yticklabels(enc_names)
axes[1].set_xlabel("embedding dim")
axes[1].set_title("Mean |activation| per embedding dim")
plt.colorbar(im, ax=axes[1])

# (3) Dead dims (std<1e-4) — capacity the encoder never uses
dead = [(pooled[n].std(0) < 1e-4).float().mean().item() for n in enc_names]
axes[2].bar(enc_names, dead, color="gray", edgecolor="black")
axes[2].set_title("Fraction of dead embedding dims (std < 1e-4)")
axes[2].tick_params(axis="x", rotation=45)
axes[2].set_ylim(0, 1)
axes[2].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

print(f"{'encoder':>13s} | {'mean|act|':>9s} | {'emb L2':>7s} | {'dead dims':>9s} | {'silence (fully padded)':>22s}")
print("-" * 80)
for name in enc_names:
    silence = (1 - valid_sample[name].float().mean().item()) if name in valid_sample else 0.0
    deadf = (pooled[name].std(0) < 1e-4).float().mean().item()
    print(
        f"{name:>13s} | {pooled[name].abs().mean().item():>9.4f} | {pooled[name].norm(dim=1).mean().item():>7.3f} | "
        f"{100 * deadf:>7.1f}% | {100 * silence:>21.1f}%"
    )
