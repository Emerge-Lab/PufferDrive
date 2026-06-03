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
# # 06 - Neural Network Architecture
# Visualize, analyze, and iterate on the DrivePolicy architecture. Covers model summary, per-encoder breakdown, forward pass shape tracing, weight distributions, and architecture comparison.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchinfo import summary
from pufferlib.ocean.drive import binding
from pufferlib.ocean.torch import Drive as DrivePolicy
from notebooks.notebook_utils import make_drive_env, zero_actions

# --- Policy architecture ---
INPUT_SIZE = 64
BACKBONE_HIDDEN_SIZE = 1024
BACKBONE_NUM_LAYERS = 3
ACTOR_HIDDEN_SIZE = 128
ACTOR_NUM_LAYERS = 3
CRITIC_HIDDEN_SIZE = 64
CRITIC_NUM_LAYERS = 2
SHARED_NETWORK = True
ENCODER_ACTIVATION = "tanh"
ENCODER_LAYER_NORM = True
BACKBONE_ACTIVATION = "gelu"
BACKBONE_LAYER_NORM = False
MASK_PADDED_FEATURES = False

env, obs, info = make_drive_env()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = DrivePolicy(
    env,
    ego_input_size=INPUT_SIZE,
    partner_input_size=INPUT_SIZE,
    lane_input_size=INPUT_SIZE,
    boundary_input_size=INPUT_SIZE,
    traffic_control_input_size=INPUT_SIZE,
    target_input_size=INPUT_SIZE,
    backbone_hidden_size=BACKBONE_HIDDEN_SIZE,
    backbone_num_layers=BACKBONE_NUM_LAYERS,
    actor_hidden_size=ACTOR_HIDDEN_SIZE,
    actor_num_layers=ACTOR_NUM_LAYERS,
    critic_hidden_size=CRITIC_HIDDEN_SIZE,
    critic_num_layers=CRITIC_NUM_LAYERS,
    encoder_activation=ENCODER_ACTIVATION,
    encoder_layer_norm=ENCODER_LAYER_NORM,
    backbone_activation=BACKBONE_ACTIVATION,
    backbone_layer_norm=BACKBONE_LAYER_NORM,
    shared_network=SHARED_NETWORK,
    mask_padded_features=MASK_PADDED_FEATURES,
).to(device)

print(f"Device: {device}")
print(f"Obs dim: {obs.shape[1]}")
print(f"Action dim: {policy.atn_dim}")
print(f"Shared network: {SHARED_NETWORK}")
print(f"Backbone: {BACKBONE_HIDDEN_SIZE} x {BACKBONE_NUM_LAYERS}L")
print(f"Actor: {ACTOR_HIDDEN_SIZE} x {ACTOR_NUM_LAYERS}L")
print(f"Critic: {CRITIC_HIDDEN_SIZE} x {CRITIC_NUM_LAYERS}L")
print(f"Encoder: {ENCODER_ACTIVATION}, LayerNorm: {ENCODER_LAYER_NORM}")

# %% [markdown]
# ## Model Summary (torchinfo)

# %%
obs_tensor = torch.FloatTensor(obs).to(device)
summary(policy, input_data=obs_tensor, depth=4, col_names=["input_size", "output_size", "num_params", "mult_adds"])

# %% [markdown]
# ## Architecture Diagram

# %%
backbone = policy.actor_backbone
cond_dim = backbone.target_dim

# Collect encoder info
encoders = [
    ("ego", env.ego_features, 1, "direct", INPUT_SIZE),
    ("conditioning", cond_dim, 1, "direct", INPUT_SIZE) if cond_dim > 0 else None,
    ("partner", env.partner_features, env.obs_slots_partners_n, "max-pool", INPUT_SIZE),
    ("lane", env.road_features, env.obs_slots_lane_kept, "max-pool", INPUT_SIZE),
    ("boundary", env.road_features, env.obs_slots_boundary_kept, "max-pool", INPUT_SIZE),
    (
        "traffic_ctrl",
        env.traffic_control_features - 2 + binding.NUM_TRAFFIC_CONTROL_TYPES + binding.NUM_TRAFFIC_CONTROL_STATES,
        env.obs_slots_traffic_controls_n,
        "max-pool (onehot)",
        INPUT_SIZE,
    ),
]
encoders = [e for e in encoders if e is not None]

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

n_enc = len(encoders)
y_positions = np.linspace(9, 1, n_enc)
colors = plt.cm.Set2(np.linspace(0, 1, n_enc))

# Draw encoders
for i, ((name, in_f, n_obj, agg, out_size), y, c) in enumerate(zip(encoders, y_positions, colors)):
    # Input box
    label = f"{name}\n{n_obj}x{in_f}" if n_obj > 1 else f"{name}\n{in_f}"
    ax.add_patch(plt.Rectangle((0.2, y - 0.3), 1.6, 0.6, facecolor=c, edgecolor="black", lw=1.2, alpha=0.8))
    ax.text(1.0, y, label, ha="center", va="center", fontsize=8, fontweight="bold")

    # Encoder box
    ax.add_patch(plt.Rectangle((2.5, y - 0.25), 2.0, 0.5, facecolor="lightyellow", edgecolor="black", lw=1))
    ax.text(3.5, y + 0.05, f"Linear({in_f},{out_size})", ha="center", va="center", fontsize=7)
    ln_label = "LN+" if ENCODER_LAYER_NORM else ""
    ax.text(
        3.5,
        y - 0.12,
        f"{ln_label}{ENCODER_ACTIVATION}+Linear({out_size},{out_size})",
        ha="center",
        va="center",
        fontsize=6,
        color="gray",
    )

    # Aggregation
    if n_obj > 1:
        ax.text(5.0, y, agg, ha="center", va="center", fontsize=7, style="italic", color="darkblue")
        arrow_start = 5.5
    else:
        arrow_start = 4.6

    # Arrows
    ax.annotate("", xy=(2.5, y), xytext=(1.8, y), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(6.0, 5.0), xytext=(arrow_start, y), arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))

# Concat box
ax.add_patch(plt.Rectangle((5.8, 4.5), 1.4, 1.0, facecolor="lightsalmon", edgecolor="black", lw=1.5))
ax.text(6.5, 5.2, "Concat", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(6.5, 4.85, f"{n_enc}x{INPUT_SIZE}={n_enc * INPUT_SIZE}", ha="center", va="center", fontsize=7)

# Backbone
ax.add_patch(plt.Rectangle((7.5, 4.5), 1.3, 1.0, facecolor="lightblue", edgecolor="black", lw=1.5))
ax.text(8.15, 5.15, f"Backbone ({BACKBONE_NUM_LAYERS}L)", ha="center", va="center", fontsize=8, fontweight="bold")
ax.text(8.15, 4.85, f"GELU+Linear\n({n_enc * INPUT_SIZE},{BACKBONE_HIDDEN_SIZE})", ha="center", va="center", fontsize=6)
ax.annotate("", xy=(7.5, 5.0), xytext=(7.2, 5.0), arrowprops=dict(arrowstyle="->", lw=1.5))

# Actor / Critic heads
ax.add_patch(plt.Rectangle((9.0, 5.7), 0.9, 0.6, facecolor="lightgreen", edgecolor="black", lw=1.2))
actor_label = f"Actor ({ACTOR_NUM_LAYERS}L)\n{BACKBONE_HIDDEN_SIZE}->{sum(policy.atn_dim)}"
if ACTOR_NUM_LAYERS > 1:
    actor_label = (
        f"Actor ({ACTOR_NUM_LAYERS}L)\n{BACKBONE_HIDDEN_SIZE}->{ACTOR_HIDDEN_SIZE}->...->{sum(policy.atn_dim)}"
    )
ax.text(9.45, 6.0, actor_label, ha="center", va="center", fontsize=6, fontweight="bold")

ax.add_patch(plt.Rectangle((9.0, 3.7), 0.9, 0.6, facecolor="plum", edgecolor="black", lw=1.2))
critic_label = f"Critic ({CRITIC_NUM_LAYERS}L)\n{BACKBONE_HIDDEN_SIZE}->1"
if CRITIC_NUM_LAYERS > 1:
    critic_label = f"Critic ({CRITIC_NUM_LAYERS}L)\n{BACKBONE_HIDDEN_SIZE}->{CRITIC_HIDDEN_SIZE}->...->1"
ax.text(9.45, 4.0, critic_label, ha="center", va="center", fontsize=6, fontweight="bold")

ax.annotate("", xy=(9.0, 6.0), xytext=(8.8, 5.3), arrowprops=dict(arrowstyle="->", lw=1.2))
ax.annotate("", xy=(9.0, 4.0), xytext=(8.8, 4.7), arrowprops=dict(arrowstyle="->", lw=1.2))

split_label = "SHARED" if SHARED_NETWORK else "SPLIT"
ax.text(8.9, 4.55, split_label, ha="center", va="center", fontsize=7, color="red", fontweight="bold")

ax.text(
    5.0,
    0.3,
    f"Encoder: {ENCODER_ACTIVATION} | LayerNorm: {ENCODER_LAYER_NORM}",
    ha="center",
    va="center",
    fontsize=8,
    color="darkgreen",
    fontweight="bold",
)

ax.set_title(
    f"DrivePolicy Architecture (encoder_size={INPUT_SIZE}, backbone={BACKBONE_HIDDEN_SIZE})",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Per-Encoder Parameter Breakdown


# %%
def count_params(module):
    return sum(p.numel() for p in module.parameters())


backbone = policy.actor_backbone
components = {
    "ego_encoder": backbone.ego_encoder,
    "lane_encoder": backbone.lane_encoder,
    "boundary_encoder": backbone.boundary_encoder,
    "partner_encoder": backbone.partner_encoder,
    "traffic_ctrl_encoder": backbone.traffic_control_encoder,
}
if backbone.target_dim > 0:
    components["target_encoder"] = backbone.target_encoder
components["backbone_mlp"] = backbone.backbone
components["actor_head"] = policy.actor_head
components["critic_head"] = policy.critic_head

names, counts = zip(*[(k, count_params(v)) for k, v in components.items()])
total = sum(counts)

print(f"{'Component':>25s} | {'Params':>10s} | {'%':>6s}")
print("-" * 48)
for n, c in zip(names, counts):
    print(f"{n:>25s} | {c:>10,d} | {c / total:>5.1%}")
print("-" * 48)
print(f"{'TOTAL':>25s} | {total:>10,d}")
if not SHARED_NETWORK:
    critic_bb = count_params(policy.critic_backbone)
    print(f"{'+ critic_backbone':>25s} | {critic_bb:>10,d}")
    print(f"{'GRAND TOTAL':>25s} | {total + critic_bb:>10,d}")

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
bars = ax.barh(names, counts, color=colors, edgecolor="black")
for bar, c in zip(bars, counts):
    ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2, f"{c:,}", va="center", fontsize=8)
ax.set_xlabel("Parameters")
ax.set_title(f"Parameter Distribution ({total:,} total)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Forward Pass Shape Trace

# %%
x = obs_tensor
backbone = policy.actor_backbone

slide_idx = env.ego_features
cond_dim = backbone.target_dim
partner_dim = env.obs_slots_partners_n * env.partner_features
lane_dim = env.obs_slots_lane_kept * env.road_features
boundary_dim = env.obs_slots_boundary_kept * env.road_features
traffic_dim = env.obs_slots_traffic_controls_n * env.traffic_control_features

# Slicing
ego_obs = x[:, :slide_idx]
slices = [("ego", 0, slide_idx, ego_obs.shape)]

if cond_dim > 0:
    cond_obs = x[:, slide_idx : slide_idx + cond_dim]
    slices.append(("conditioning", slide_idx, slide_idx + cond_dim, cond_obs.shape))
    slide_idx += cond_dim

partner_obs = x[:, slide_idx : slide_idx + partner_dim]
slices.append(("partners", slide_idx, slide_idx + partner_dim, partner_obs.shape))
slide_idx += partner_dim

lane_obs = x[:, slide_idx : slide_idx + lane_dim]
slices.append(("lanes", slide_idx, slide_idx + lane_dim, lane_obs.shape))
slide_idx += lane_dim

boundary_obs = x[:, slide_idx : slide_idx + boundary_dim]
slices.append(("boundaries", slide_idx, slide_idx + boundary_dim, boundary_obs.shape))
slide_idx += boundary_dim

traffic_obs = x[:, slide_idx : slide_idx + traffic_dim]
slices.append(("traffic_ctrl", slide_idx, slide_idx + traffic_dim, traffic_obs.shape))

print(f"Obs buffer layout (total={x.shape[1]}):")
print(f"{'Name':>15s} | {'Start':>5s} | {'End':>5s} | {'Width':>5s} | Shape")
print("-" * 65)
for name, start, end, shape in slices:
    print(f"{name:>15s} | {start:>5d} | {end:>5d} | {end - start:>5d} | {shape}")

# Forward through encoders
print("\nEncoder outputs:")
with torch.no_grad():
    ego_enc = backbone.ego_encoder(ego_obs)
    print(f"  ego_encoder:     {ego_obs.shape} -> {ego_enc.shape}")

    if cond_dim > 0:
        cond_enc = backbone.target_encoder(cond_obs)
        print(f"  cond_encoder:    {cond_obs.shape} -> {cond_enc.shape}")

    p_reshaped = partner_obs.view(-1, env.obs_slots_partners_n, env.partner_features)
    p_enc, _ = backbone.partner_encoder(p_reshaped).max(dim=1)
    print(f"  partner_encoder: {partner_obs.shape} -> view {p_reshaped.shape} -> encode -> max-pool -> {p_enc.shape}")

    l_reshaped = lane_obs.view(-1, env.obs_slots_lane_kept, env.road_features)
    l_enc, _ = backbone.lane_encoder(l_reshaped).max(dim=1)
    print(f"  lane_encoder:    {lane_obs.shape} -> view {l_reshaped.shape} -> encode -> max-pool -> {l_enc.shape}")

    b_reshaped = boundary_obs.view(-1, env.obs_slots_boundary_kept, env.road_features)
    b_enc, _ = backbone.boundary_encoder(b_reshaped).max(dim=1)
    print(f"  bound_encoder:   {boundary_obs.shape} -> view {b_reshaped.shape} -> encode -> max-pool -> {b_enc.shape}")

    t_reshaped = traffic_obs.view(-1, env.obs_slots_traffic_controls_n, env.traffic_control_features)
    t_cont = t_reshaped[:, :, : env.traffic_control_features - 2]
    t_type = t_reshaped[:, :, env.traffic_control_features - 2]
    t_state = t_reshaped[:, :, env.traffic_control_features - 1]
    t_type_onehot = F.one_hot(t_type.long(), num_classes=binding.NUM_TRAFFIC_CONTROL_TYPES).float()
    t_state_onehot = F.one_hot(t_state.long(), num_classes=binding.NUM_TRAFFIC_CONTROL_STATES).float()
    t_input = torch.cat([t_cont, t_type_onehot, t_state_onehot], dim=2)
    t_enc, _ = backbone.traffic_control_encoder(t_input).max(dim=1)
    print(
        f"  traffic_encoder: {traffic_obs.shape} -> view {t_reshaped.shape} -> onehot {t_input.shape} -> encode -> max-pool -> {t_enc.shape}"
    )

    # Concat + backbone
    features = [ego_enc, l_enc, b_enc, p_enc, t_enc]
    if cond_dim > 0:
        features.append(cond_enc)
    concat = torch.cat(features, dim=1)
    hidden = backbone.backbone(concat)
    print(f"\n  concat: {concat.shape}")
    print(f"  backbone_mlp: {concat.shape} -> {hidden.shape}")

    # Heads
    actor_out = policy.actor_head(hidden)
    critic_out = policy.critic_head(hidden)
    print(f"  actor_head:  {hidden.shape} -> {actor_out.shape} (split into {policy.atn_dim})")
    print(f"  critic_head: {hidden.shape} -> {critic_out.shape}")

# %% [markdown]
# ## Weight Distributions by Layer

# %%
weight_data = [
    (n, p.data.cpu().numpy().flatten()) for n, p in policy.named_parameters() if "weight" in n and p.dim() >= 2
]

n_weights = len(weight_data)
cols = 4
rows = (n_weights + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
axes = axes.flatten()

for i, (name, w) in enumerate(weight_data):
    ax = axes[i]
    ax.hist(w, bins=50, edgecolor="black", alpha=0.7, density=True)
    ax.set_title(name.replace("actor_backbone.", ""), fontsize=7)
    ax.axvline(0, color="red", ls="--", lw=0.5)
    ax.text(0.95, 0.95, f"std={w.std():.3f}", transform=ax.transAxes, fontsize=6, ha="right", va="top")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle("Weight Distributions (init)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Activation Analysis (per encoder)

# %%
policy.eval()
with torch.no_grad():
    hidden = policy.actor_backbone(obs_tensor, env.ego_features)
    action_logits, value = policy.decode_actions(hidden)

# Collect per-encoder activations
activations = {}
with torch.no_grad():
    slide = env.ego_features
    activations["ego"] = backbone.ego_encoder(obs_tensor[:, : env.ego_features])

    if cond_dim > 0:
        activations["conditioning"] = backbone.target_encoder(obs_tensor[:, slide : slide + cond_dim])
        slide += cond_dim

    p_obs = obs_tensor[:, slide : slide + partner_dim].view(-1, env.obs_slots_partners_n, env.partner_features)
    activations["partner"], _ = backbone.partner_encoder(p_obs).max(dim=1)
    slide += partner_dim

    l_obs = obs_tensor[:, slide : slide + lane_dim].view(-1, env.obs_slots_lane_kept, env.road_features)
    activations["lane"], _ = backbone.lane_encoder(l_obs).max(dim=1)
    slide += lane_dim

    b_obs = obs_tensor[:, slide : slide + boundary_dim].view(-1, env.obs_slots_boundary_kept, env.road_features)
    activations["boundary"], _ = backbone.boundary_encoder(b_obs).max(dim=1)
    slide += boundary_dim

    t_obs = obs_tensor[:, slide : slide + traffic_dim].view(
        -1, env.obs_slots_traffic_controls_n, env.traffic_control_features
    )
    t_cont = t_obs[:, :, : env.traffic_control_features - 2]
    t_type = t_obs[:, :, env.traffic_control_features - 2]
    t_state = t_obs[:, :, env.traffic_control_features - 1]
    t_type_onehot = F.one_hot(t_type.long(), num_classes=binding.NUM_TRAFFIC_CONTROL_TYPES).float()
    t_state_onehot = F.one_hot(t_state.long(), num_classes=binding.NUM_TRAFFIC_CONTROL_STATES).float()
    t_input = torch.cat([t_cont, t_type_onehot, t_state_onehot], dim=2)
    activations["traffic_ctrl"], _ = backbone.traffic_control_encoder(t_input).max(dim=1)

    activations["hidden"] = hidden

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
axes = axes.flatten()
for i, (name, act) in enumerate(activations.items()):
    if i >= len(axes):
        break
    vals = act.cpu().numpy().flatten()
    ax = axes[i]
    ax.hist(vals, bins=50, edgecolor="black", alpha=0.7)
    dead = (act.abs().sum(dim=0) == 0).sum().item()
    ax.set_title(f"{name} (dead={dead}/{act.shape[1]})", fontsize=9)
    ax.text(
        0.95,
        0.95,
        f"mean={vals.mean():.3f}\nstd={vals.std():.3f}",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="top",
    )

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle("Per-Encoder Activation Distributions", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Encoder Embedding Similarity (cosine)

# %%
# Mean embedding per encoder (exclude hidden — different dim)
emb_names = [k for k in activations.keys() if k != "hidden"]
emb_means = torch.stack([activations[k].mean(dim=0) for k in emb_names])
emb_norm = F.normalize(emb_means, dim=1)
sim_matrix = (emb_norm @ emb_norm.T).cpu().numpy()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(emb_names)))
ax.set_yticks(range(len(emb_names)))
ax.set_xticklabels(emb_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(emb_names, fontsize=8)
for i in range(len(emb_names)):
    for j in range(len(emb_names)):
        ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax)
ax.set_title("Cosine Similarity Between Encoder Mean Embeddings")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Architecture Comparison
# Compare different architecture configs side-by-side without training.

# %%
configs = [
    {"name": "tiny", "encoder_size": 32, "backbone_hidden_size": 64},
    {"name": "small", "encoder_size": 64, "backbone_hidden_size": 128},
    {"name": "medium", "encoder_size": 128, "backbone_hidden_size": 256, "backbone_num_layers": 2},
    {
        "name": "large",
        "encoder_size": 128,
        "backbone_hidden_size": 512,
        "backbone_num_layers": 2,
        "actor_num_layers": 2,
        "actor_hidden_size": 256,
        "critic_num_layers": 2,
        "critic_hidden_size": 256,
    },
    {
        "name": "xlarge",
        "encoder_size": 256,
        "backbone_hidden_size": 1024,
        "backbone_num_layers": 3,
        "actor_num_layers": 2,
        "actor_hidden_size": 512,
        "critic_num_layers": 2,
        "critic_hidden_size": 512,
    },
    {"name": "small+tanh", "encoder_size": 64, "backbone_hidden_size": 128, "encoder_activation": "tanh"},
    {
        "name": "medium+tanh",
        "encoder_size": 128,
        "backbone_hidden_size": 256,
        "backbone_num_layers": 2,
        "encoder_activation": "tanh",
    },
]

POLICY_DEFAULTS = {
    "ego_input_size": 64,
    "partner_input_size": 64,
    "lane_input_size": 64,
    "boundary_input_size": 64,
    "traffic_control_input_size": 64,
    "target_input_size": 64,
    "backbone_num_layers": 1,
    "actor_hidden_size": 128,
    "actor_num_layers": 0,
    "critic_hidden_size": 128,
    "critic_num_layers": 0,
    "encoder_activation": "relu",
    "encoder_layer_norm": True,
    "backbone_activation": "gelu",
    "backbone_layer_norm": False,
    "shared_network": True,
    "mask_padded_features": False,
}

results = []
for cfg in configs:
    name = cfg["name"]
    encoder_size = cfg.get("encoder_size", POLICY_DEFAULTS["ego_input_size"])
    full_cfg = {**POLICY_DEFAULTS, **{k: v for k, v in cfg.items() if k not in ("name", "encoder_size")}}
    full_cfg.update(
        {
            "ego_input_size": encoder_size,
            "partner_input_size": encoder_size,
            "lane_input_size": encoder_size,
            "boundary_input_size": encoder_size,
            "traffic_control_input_size": encoder_size,
            "target_input_size": encoder_size,
        }
    )
    p = DrivePolicy(env, **full_cfg).to(device)
    n_params = sum(pp.numel() for pp in p.parameters())

    with torch.no_grad():
        import time

        t0 = time.time()
        for _ in range(100):
            p(obs_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms_per_fwd = (time.time() - t0) / 100 * 1000

    results.append({"name": name, "encoder_size": encoder_size, "params": n_params, "ms/fwd": ms_per_fwd, **full_cfg})
    del p

print(
    f"{'Config':>12s} | {'enc':>5s} | {'bb_h':>5s} | {'bb_L':>4s} | {'act_h':>5s} | {'act_L':>5s} | {'crt_h':>5s} | {'crt_L':>5s} | {'enc_act':>7s} | {'Params':>10s} | {'ms/fwd':>8s}"
)
print("-" * 105)
for r in results:
    print(
        f"{r['name']:>12s} | {r['encoder_size']:>5d} | {r['backbone_hidden_size']:>5d} | {r['backbone_num_layers']:>4d} | {r['actor_hidden_size']:>5d} | {r['actor_num_layers']:>5d} | {r['critic_hidden_size']:>5d} | {r['critic_num_layers']:>5d} | {r['encoder_activation']:>7s} | {r['params']:>10,d} | {r['ms/fwd']:>7.2f}ms"
    )

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
names = [r["name"] for r in results]
params = [r["params"] for r in results]
times = [r["ms/fwd"] for r in results]

bar_colors = ["coral" if r["encoder_activation"] == "tanh" else "steelblue" for r in results]

axes[0].bar(names, params, color=bar_colors, edgecolor="black")
axes[0].set_ylabel("Parameters")
axes[0].set_title("Parameter Count (orange=tanh encoder)")
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(params):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)

axes[1].bar(names, times, color=bar_colors, edgecolor="black")
axes[1].set_ylabel("ms / forward")
axes[1].set_title(f"Forward Pass Latency ({env.num_agents} agents)")
axes[1].tick_params(axis="x", rotation=30)
for i, v in enumerate(times):
    axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Observation Buffer Utilization
# How much of each observation slot is actually filled (non-zero)?

# %%
# Run a few steps to get diverse observations
actions = zero_actions(env)
all_obs = [obs]
for _ in range(20):
    o, _, _, _, _ = env.step(actions)
    all_obs.append(o)
stacked = np.concatenate(all_obs, axis=0)

slide = env.ego_features
segments = [("ego", 0, env.ego_features, 1, env.ego_features)]
if cond_dim > 0:
    segments.append(("conditioning", slide, slide + cond_dim, 1, cond_dim))
    slide += cond_dim
segments.append(("partners", slide, slide + partner_dim, env.obs_slots_partners_n, env.partner_features))
slide += partner_dim
segments.append(("lanes", slide, slide + lane_dim, env.obs_slots_lane_kept, env.road_features))
slide += lane_dim
segments.append(("boundaries", slide, slide + boundary_dim, env.obs_slots_boundary_kept, env.road_features))
slide += boundary_dim
segments.append(("traffic", slide, slide + traffic_dim, env.obs_slots_traffic_controls_n, env.traffic_control_features))

print(f"{'Segment':>15s} | {'Slots':>5s} | {'Features':>8s} | {'Fill %':>7s} | {'Mean':>8s} | {'Std':>8s}")
print("-" * 65)
fill_rates = []
seg_names = []
for name, start, end, n_slots, n_feat in segments:
    chunk = stacked[:, start:end]
    if n_slots > 1:
        reshaped = chunk.reshape(-1, n_slots, n_feat)
        # A slot is "filled" if any feature is non-zero
        filled = (np.abs(reshaped).sum(axis=2) > 1e-8).mean()
    else:
        filled = (np.abs(chunk) > 1e-8).mean()
    fill_rates.append(filled * 100)
    seg_names.append(name)
    print(f"{name:>15s} | {n_slots:>5d} | {n_feat:>8d} | {filled:>6.1%} | {chunk.mean():>8.4f} | {chunk.std():>8.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#2ecc71" if f > 50 else "#e74c3c" if f < 10 else "#f39c12" for f in fill_rates]
ax.barh(seg_names, fill_rates, color=colors, edgecolor="black")
ax.set_xlabel("Fill Rate (%)")
ax.set_title("Observation Slot Utilization")
ax.axvline(50, color="gray", ls="--", alpha=0.5)
for i, v in enumerate(fill_rates):
    ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Effective Receptive Field
# Which input features have the most influence on the hidden representation?

# %%
# Jacobian-based sensitivity: d(hidden) / d(obs) magnitude
sample = obs_tensor[:1].clone().requires_grad_(True)
hidden = policy.actor_backbone(sample, env.ego_features)
# Sum hidden to scalar for backward
hidden.sum().backward()
sensitivity = sample.grad.abs().squeeze().cpu().numpy()

fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [2, 1]})

# Full sensitivity
axes[0].plot(sensitivity, lw=0.5, color="steelblue")
axes[0].set_ylabel("|grad|")
axes[0].set_title("Input Feature Sensitivity (|d hidden / d obs|)")

# Mark segments
seg_boundaries = [0, env.ego_features]
seg_labels = ["ego"]
s = env.ego_features
if cond_dim > 0:
    s += cond_dim
    seg_boundaries.append(s)
    seg_labels.append("cond")
for name, dim in [
    ("partners", partner_dim),
    ("lanes", lane_dim),
    ("boundaries", boundary_dim),
    ("traffic", traffic_dim),
]:
    s += dim
    seg_boundaries.append(s)
    seg_labels.append(name)

seg_colors = plt.cm.Set2(np.linspace(0, 1, len(seg_labels)))
for i, (label, c) in enumerate(zip(seg_labels, seg_colors)):
    start, end = seg_boundaries[i], seg_boundaries[i + 1]
    axes[0].axvspan(start, end, alpha=0.15, color=c)
    axes[0].text((start + end) / 2, axes[0].get_ylim()[1] * 0.9, label, ha="center", fontsize=7, color="black")

# Per-segment mean sensitivity
seg_means = []
for i in range(len(seg_labels)):
    start, end = seg_boundaries[i], seg_boundaries[i + 1]
    seg_means.append(sensitivity[start:end].mean())

axes[1].bar(seg_labels, seg_means, color=seg_colors, edgecolor="black")
axes[1].set_ylabel("Mean |grad|")
axes[1].set_title("Mean Sensitivity per Observation Segment")

plt.tight_layout()
plt.show()

policy.zero_grad()
