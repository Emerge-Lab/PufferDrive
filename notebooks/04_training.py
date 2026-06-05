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
# # 04 - RL Training Loop Debug
# End-to-end data flow from env -> policy -> loss. Debug encoding, sampling, advantages, gradients.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from notebooks.notebook_utils import make_drive_env, make_drive_policy, zero_actions

env, obs, info = make_drive_env()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = make_drive_policy(env, device)
print(f"Policy on {device}, params: {sum(p.numel() for p in policy.parameters()):,}")
print(f"Action dim: {policy.atn_dim}, act_shape: {(env.num_agents, len(env.single_action_space.nvec))}")

# %% [markdown]
# ### Optional: load checkpoint

# %%
# CHECKPOINT_PATH = ''
# state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
# state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# print('Checkpoint loaded')

# %% [markdown]
# ## Encode observations

# %%
actions = zero_actions(env)
obs, rew, term, trunc, info = env.step(actions)

obs_tensor = torch.FloatTensor(obs).to(device)
with torch.no_grad():
    hidden = policy.encode_observations(obs_tensor)

print(f"Hidden shape: {hidden.shape}")
print(f"Hidden stats: min={hidden.min():.4f}, max={hidden.max():.4f}, mean={hidden.mean():.4f}")
print(f"NaN in hidden: {torch.isnan(hidden).sum().item()}")
print(f"Dead neurons (always 0): {(hidden.abs().sum(dim=0) == 0).sum().item()}/{hidden.shape[1]}")
print(f"% near-zero (<1e-6): {(hidden.abs() < 1e-6).float().mean().item() * 100:.1f}%")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(hidden.cpu().numpy().flatten(), bins=50, edgecolor="black", alpha=0.7)
ax.set_title("Hidden activation distribution")
ax.set_xlabel("Activation value")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Action sampling

# %%
with torch.no_grad():
    action_logits, value = policy.decode_actions(hidden)

for i, logit in enumerate(action_logits):
    print(f"Action head {i}: shape={logit.shape}")
    probs = F.softmax(logit, dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    max_entropy = np.log(logit.shape[-1])
    print(f"  Entropy: {entropy:.4f} / {max_entropy:.4f} (max) = {entropy / max_entropy:.2%}")
    print(f"  Logit range: [{logit.min():.3f}, {logit.max():.3f}]")

print(f"\nValue: mean={value.mean():.4f}, std={value.std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
probs = F.softmax(action_logits[0], dim=-1)
mean_probs = probs.mean(dim=0).cpu().numpy()
axes[0].bar(range(len(mean_probs)), mean_probs, edgecolor="black", alpha=0.7)
axes[0].axhline(1.0 / len(mean_probs), color="red", ls="--", label="uniform")
axes[0].set_xlabel("Action")
axes[0].set_ylabel("Probability")
axes[0].set_title("Mean action probabilities")
axes[0].legend()

axes[1].hist(value.cpu().numpy().flatten(), bins=20, edgecolor="black", alpha=0.7, color="purple")
axes[1].set_title("Value predictions")
axes[1].set_xlabel("Value")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Manual encode trace: check each encoder for NaN

# %%
x = obs_tensor
backbone = policy.actor_backbone
slide_idx = env.ego_features

ego_obs = x[:, :slide_idx]
print(
    f"ego_obs: shape={ego_obs.shape}, NaN={torch.isnan(ego_obs).sum().item()}, range=[{ego_obs.min():.3f}, {ego_obs.max():.3f}]"
)

context_dim = backbone.context_dim
if context_dim > 0:
    context_obs = x[:, slide_idx : slide_idx + context_dim]
    slide_idx += context_dim
    print(f"context_obs: shape={context_obs.shape}, NaN={torch.isnan(context_obs).sum().item()}")

partner_dim = env.obs_slots_partners_n * env.partner_features
lane_dim = env.obs_slots_lane_kept * env.road_features
boundary_dim = env.obs_slots_boundary_kept * env.road_features

partner_obs = x[:, slide_idx : slide_idx + partner_dim]
slide_idx += partner_dim
lane_obs = x[:, slide_idx : slide_idx + lane_dim]
slide_idx += lane_dim
boundary_obs = x[:, slide_idx : slide_idx + boundary_dim]
slide_idx += boundary_dim

with torch.no_grad():
    ego_enc = backbone.ego_encoder(ego_obs)
    partner_enc, _ = backbone.partner_encoder(partner_obs.view(-1, env.obs_slots_partners_n, env.partner_features)).max(
        dim=1
    )
    lane_enc, _ = backbone.lane_encoder(lane_obs.view(-1, env.obs_slots_lane_kept, env.road_features)).max(dim=1)
    bound_enc, _ = backbone.boundary_encoder(boundary_obs.view(-1, env.obs_slots_boundary_kept, env.road_features)).max(
        dim=1
    )

for name, enc in [("ego", ego_enc), ("partner", partner_enc), ("lane", lane_enc), ("boundary", bound_enc)]:
    print(
        f"{name:>10s}_enc: NaN={torch.isnan(enc).sum().item()}, dead={((enc.abs().sum(dim=0) == 0).sum().item())}, range=[{enc.min():.3f}, {enc.max():.3f}]"
    )

if context_dim > 0:
    with torch.no_grad():
        context_enc = backbone.context_encoder(context_obs)
    print(
        f"{'context':>10s}_enc: NaN={torch.isnan(context_enc).sum().item()}, dead={((context_enc.abs().sum(dim=0) == 0).sum().item())}, range=[{context_enc.min():.3f}, {context_enc.max():.3f}]"
    )

# %% [markdown]
# ## Forward-backward: fake advantage, loss, grads

# %%
policy.train()
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

action_logits_list, value = policy(obs_tensor)

fake_actions = torch.randint(0, env.single_action_space.nvec[0], (env.num_agents,), device=device)
fake_advantages = torch.randn(env.num_agents, device=device)
fake_returns = torch.randn(env.num_agents, device=device)
fake_old_logprobs = torch.randn(env.num_agents, device=device)

logits = action_logits_list[0]
dist = torch.distributions.Categorical(logits=logits)
new_logprobs = dist.log_prob(fake_actions)
entropy = dist.entropy()

ratio = torch.exp(new_logprobs - fake_old_logprobs)
clip_coef = 0.2
pg_loss1 = -fake_advantages * ratio
pg_loss2 = -fake_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
v_loss = 0.5 * ((value.squeeze() - fake_returns) ** 2).mean()
entropy_loss = entropy.mean()
loss = pg_loss + 0.5 * v_loss - 0.01 * entropy_loss

print(f"pg_loss: {pg_loss.item():.4f}")
print(f"v_loss:  {v_loss.item():.4f}")
print(f"entropy: {entropy_loss.item():.4f}")
print(f"total:   {loss.item():.4f}")
print(f"ratio: mean={ratio.mean():.4f}, std={ratio.std():.4f}")

optimizer.zero_grad()
loss.backward()
total_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))
print(f"\nTotal grad norm: {total_grad_norm:.4f}")
print(f"NaN in loss: {torch.isnan(loss).item()}")

# %% [markdown]
# ## Gradient flow: per-parameter analysis

# %%
print(f"{'Parameter':>45s} | {'shape':>20s} | {'grad_norm':>10s} {'grad_mean':>10s} {'grad_max':>10s} | flag")
print("-" * 120)
for name, param in policy.named_parameters():
    if param.grad is not None:
        g = param.grad
        norm = g.norm().item()
        mean = g.mean().item()
        mx = g.abs().max().item()
        flag = ""
        if norm == 0:
            flag = "ZERO GRAD"
        elif norm > 100:
            flag = "EXPLODING"
        elif norm < 1e-7:
            flag = "VANISHING"
        print(f"{name:>45s} | {str(list(param.shape)):>20s} | {norm:10.6f} {mean:10.6f} {mx:10.6f} | {flag}")
    else:
        print(f"{name:>45s} | {str(list(param.shape)):>20s} | NO GRAD")

# %% [markdown]
# ## Experience buffer simulation: 128-step rollout

# %%
HORIZON = 128
obs_dim = obs.shape[1]

obs_buf = np.zeros((HORIZON, env.num_agents, obs_dim), dtype=np.float32)
act_buf = np.zeros((HORIZON, env.num_agents), dtype=np.int64)
rew_buf = np.zeros((HORIZON, env.num_agents), dtype=np.float32)
val_buf = np.zeros((HORIZON, env.num_agents), dtype=np.float32)
logp_buf = np.zeros((HORIZON, env.num_agents), dtype=np.float32)
done_buf = np.zeros((HORIZON, env.num_agents), dtype=np.float32)

policy.eval()
for t in range(HORIZON):
    obs_t = torch.FloatTensor(obs).to(device)
    with torch.no_grad():
        logits_list, val = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits_list[0])
        act = dist.sample()
        logp = dist.log_prob(act)

    obs_buf[t] = obs
    act_buf[t] = act.cpu().numpy()
    val_buf[t] = val.squeeze().cpu().numpy()
    logp_buf[t] = logp.cpu().numpy()

    # Reshape (N,) -> (N, 1) for env.step with MultiDiscrete
    env_actions = act.cpu().numpy().reshape(env.num_agents, len(env.single_action_space.nvec))
    obs, rew, term, trunc, info = env.step(env_actions)
    rew_buf[t] = rew
    done_buf[t] = term | trunc

print(f"Buffer shapes: obs={obs_buf.shape}, act={act_buf.shape}, rew={rew_buf.shape}")
print(f"Reward stats: mean={rew_buf.mean():.5f}, std={rew_buf.std():.5f}")
print(f"Value stats: mean={val_buf.mean():.5f}, std={val_buf.std():.5f}")
print(f"Done count: {done_buf.sum():.0f}")
print(f"LogProb stats: mean={logp_buf.mean():.4f}, std={logp_buf.std():.4f}")

# %% [markdown]
# ## GAE advantage computation

# %%
gamma, lam = 0.98, 0.95
advantages = np.zeros_like(rew_buf)

last_gae = np.zeros(env.num_agents)
for t in reversed(range(HORIZON - 1)):
    next_non_terminal = 1.0 - done_buf[t + 1]
    delta = rew_buf[t + 1] + gamma * val_buf[t + 1] * next_non_terminal - val_buf[t]
    last_gae = delta + gamma * lam * last_gae * next_non_terminal
    advantages[t] = last_gae

returns = advantages + val_buf

print(f"Advantages: mean={advantages.mean():.5f}, std={advantages.std():.5f}")
print(f"Returns: mean={returns.mean():.5f}, std={returns.std():.5f}")
print(f"Advantage vs Return corr: {np.corrcoef(advantages.flatten(), returns.flatten())[0, 1]:.4f}")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].hist(advantages.flatten(), bins=50, edgecolor="black", alpha=0.7)
axes[0].set_title(f"Advantage distribution (std={advantages.std():.4f})")

axes[1].hist(returns.flatten(), bins=50, edgecolor="black", alpha=0.7, color="orange")
axes[1].set_title("Returns distribution")

axes[2].plot(advantages.mean(axis=1))
axes[2].set_xlabel("Step")
axes[2].set_ylabel("Mean advantage")
axes[2].set_title("Mean advantage over time")

axes[3].plot(done_buf.mean(axis=1), color="orange")
axes[3].set_xlabel("Step")
axes[3].set_ylabel("Mean done")
axes[3].set_title("Mean done over time")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## PPO loss components

# %%
MB = 16
mb_obs = torch.FloatTensor(obs_buf[:MB].reshape(-1, obs_dim)).to(device)
mb_act = torch.LongTensor(act_buf[:MB].flatten()).to(device)
mb_old_logp = torch.FloatTensor(logp_buf[:MB].flatten()).to(device)
mb_adv = torch.FloatTensor(advantages[:MB].flatten()).to(device)
mb_ret = torch.FloatTensor(returns[:MB].flatten()).to(device)
mb_old_val = torch.FloatTensor(val_buf[:MB].flatten()).to(device)

mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

policy.train()
logits_list, newvalue = policy(mb_obs)
newvalue = newvalue.squeeze()
dist = torch.distributions.Categorical(logits=logits_list[0])
new_logp = dist.log_prob(mb_act)
entropy = dist.entropy()

ratio = torch.exp(new_logp - mb_old_logp)
print(f"Ratio: mean={ratio.mean():.4f}, std={ratio.std():.4f}, min={ratio.min():.4f}, max={ratio.max():.4f}")
if ratio.mean() < 0.5 or ratio.mean() > 2.0:
    print("WARNING: ratio far from 1.0, policy may have diverged")

clip_coef = 0.2
pg_loss1 = -mb_adv * ratio
pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()

vf_clip = 0.2
v_clipped = mb_old_val + torch.clamp(newvalue - mb_old_val, -vf_clip, vf_clip)
v_loss_unclipped = (newvalue - mb_ret) ** 2
v_loss_clipped = (v_clipped - mb_ret) ** 2
v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

entropy_loss = entropy.mean()

print(f"\npg_loss:  {pg_loss.item():.6f}")
print(f"v_loss:   {v_loss.item():.6f}")
print(f"entropy:  {entropy_loss.item():.6f} (max={np.log(env.single_action_space.nvec[0]):.4f})")
print(f"total:    {(pg_loss + 0.5 * v_loss - 0.01 * entropy_loss).item():.6f}")

# %% [markdown]
# ## 5-epoch sanity training

# %%
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
all_obs = torch.FloatTensor(obs_buf.reshape(-1, obs_dim)).to(device)
all_act = torch.LongTensor(act_buf.flatten()).to(device)
all_old_logp = torch.FloatTensor(logp_buf.flatten()).to(device)
all_adv = torch.FloatTensor(advantages.flatten()).to(device)
all_ret = torch.FloatTensor(returns.flatten()).to(device)

all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

N_EPOCHS = 5
history = {"pg_loss": [], "v_loss": [], "entropy": [], "kl": []}

policy.train()
for epoch in range(N_EPOCHS):
    logits_list, newval = policy(all_obs)
    newval = newval.squeeze()
    dist = torch.distributions.Categorical(logits=logits_list[0])
    new_logp = dist.log_prob(all_act)
    ent = dist.entropy().mean()

    ratio = torch.exp(new_logp - all_old_logp)
    approx_kl = (all_old_logp - new_logp).mean()

    pg1 = -all_adv * ratio
    pg2 = -all_adv * torch.clamp(ratio, 0.8, 1.2)
    pg = torch.max(pg1, pg2).mean()
    vl = 0.5 * ((newval - all_ret) ** 2).mean()
    loss = pg + 0.5 * vl - 0.01 * ent

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()

    history["pg_loss"].append(pg.item())
    history["v_loss"].append(vl.item())
    history["entropy"].append(ent.item())
    history["kl"].append(approx_kl.item())
    print(f"Epoch {epoch}: pg={pg.item():.5f}, v={vl.item():.5f}, ent={ent.item():.4f}, kl={approx_kl.item():.5f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for i, (key, color) in enumerate(zip(history.keys(), ["red", "blue", "green", "orange"])):
    axes[i].plot(history[key], "-o", color=color)
    axes[i].set_title(key)
    axes[i].set_xlabel("Epoch")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Value accuracy: predicted vs actual returns

# %%
policy.eval()
with torch.no_grad():
    _, pred_values = policy(all_obs)
pred_values = pred_values.squeeze().cpu().numpy()
actual_returns = returns.flatten()

var_actual = np.var(actual_returns)
explained_var = 1 - np.var(actual_returns - pred_values) / (var_actual + 1e-8) if var_actual > 1e-8 else 0.0

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(actual_returns, pred_values, alpha=0.3, s=10)
lims = [min(actual_returns.min(), pred_values.min()), max(actual_returns.max(), pred_values.max())]
ax.plot(lims, lims, "r--", label="perfect")
ax.set_xlabel("Actual return")
ax.set_ylabel("Predicted value")
ax.set_title(f"Value accuracy (explained var: {explained_var:.4f})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Explained variance: {explained_var:.4f}")
print(f"Value MSE: {np.mean((actual_returns - pred_values) ** 2):.6f}")
if explained_var < 0:
    print("WARNING: negative explained variance, value head worse than predicting mean")
