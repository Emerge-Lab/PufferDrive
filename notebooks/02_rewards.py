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
# # 02 - Reward Signals Debug
# Understand reward magnitudes, components, and correlation with agent behavior.

# %%
import numpy as np
import matplotlib.pyplot as plt
from notebooks.notebook_utils import COEF_NAMES, make_drive_env, random_actions, zero_actions

env, obs, info = make_drive_env()

print(
    f"env ready: {env.num_agents} agents, obs={obs.shape}, act_shape={(env.num_agents, len(env.single_action_space.nvec))}"
)
print(
    f"ego_features={env.ego_features}, num_reward_coefs={env.num_reward_coefs}, obs_slots_partners_n={env.obs_slots_partners_n}, partner_features={env.partner_features}"
)
print(
    f"obs_slots_lane_kept={env.obs_slots_lane_kept}, obs_slots_boundary_kept={env.obs_slots_boundary_kept}, road_features={env.road_features}"
)
print(
    f"obs_slots_traffic_controls_n={env.obs_slots_traffic_controls_n}, traffic_control_features={env.traffic_control_features}"
)

# %% [markdown]
# ## Single step: no-op reward distribution

# %%
actions = zero_actions(env)
obs, rew, term, trunc, info = env.step(actions)

print(f"reward shape: {rew.shape}")
print(f"min: {rew.min():.6f}, max: {rew.max():.6f}, mean: {rew.mean():.6f}, std: {rew.std():.6f}")
print(f"NaN: {np.isnan(rew).sum()}, all zero: {(rew == 0).all()}")
print(f"terminals: {term.sum()}, truncations: {trunc.sum()}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(rew)), rew, color=["red" if r < 0 else "green" for r in rew])
ax.set_xlabel("Agent index")
ax.set_ylabel("Reward")
ax.set_title("Single step reward (no-op action)")
ax.axhline(0, color="black", lw=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 100-step rollout: reward heatmap and cumulative returns

# %%
N_STEPS = 100
rewards_history = np.zeros((N_STEPS, env.num_agents))
terms_history = np.zeros((N_STEPS, env.num_agents))

for t in range(N_STEPS):
    actions = random_actions(env)
    obs, rew, term, trunc, info = env.step(actions)
    rewards_history[t] = rew
    terms_history[t] = term

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(rewards_history.mean(axis=1))
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Mean reward")
axes[0].set_title("Mean reward per step")

im = axes[1].imshow(rewards_history.T, aspect="auto", cmap="RdYlGn", interpolation="nearest")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Agent")
axes[1].set_title("Reward heatmap (steps x agents)")
plt.colorbar(im, ax=axes[1])

cum_returns = rewards_history.cumsum(axis=0)
for i in range(min(8, env.num_agents)):
    axes[2].plot(cum_returns[:, i], alpha=0.6, label=f"agent {i}")
axes[2].set_xlabel("Step")
axes[2].set_ylabel("Cumulative return")
axes[2].set_title("Cumulative returns")
axes[2].legend(fontsize=7)
plt.tight_layout()
plt.show()

print(f"Total reward stats: mean={rewards_history.mean():.5f}, std={rewards_history.std():.5f}")
print(f"Per-episode return (100 steps): mean={cum_returns[-1].mean():.3f}, std={cum_returns[-1].std():.3f}")

# %% [markdown]
# ## Reward coefficient inspection

# %%
all_coefs = obs[:, env.ego_features : env.ego_features + env.num_reward_coefs]
print(f"Reward coefs shape: {all_coefs.shape}")
print()
print(f"{'Coef':>15s} | {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
print("-" * 55)
for i, name in enumerate(COEF_NAMES):
    c = all_coefs[:, i]
    print(f"{name:>15s} | {c.mean():8.4f} {c.std():8.4f} {c.min():8.4f} {c.max():8.4f}")

# %% [markdown]
# ## Terminal analysis

# %%
N_STEPS = 200
term_steps, trunc_steps = [], []
term_rewards, trunc_rewards = [], []

for t in range(N_STEPS):
    actions = random_actions(env)
    obs, rew, term, trunc, info = env.step(actions)
    for i in range(env.num_agents):
        if term[i]:
            term_steps.append(t)
            term_rewards.append(rew[i])
        if trunc[i]:
            trunc_steps.append(t)
            trunc_rewards.append(rew[i])

print(f"Terminals: {len(term_steps)}, Truncations: {len(trunc_steps)}")
if term_rewards:
    tr = np.array(term_rewards)
    print(f"Terminal reward: mean={tr.mean():.4f}, std={tr.std():.4f}")
    n_positive = (tr > 0).sum()
    n_negative = (tr < 0).sum()
    n_zero = (tr == 0).sum()
    print(f"  positive: {n_positive}, negative: {n_negative}, zero: {n_zero}")

fig, ax = plt.subplots(figsize=(10, 4))
if term_steps:
    ax.scatter(term_steps, term_rewards, c="red", s=20, alpha=0.5, label=f"terminal ({len(term_steps)})")
if trunc_steps:
    ax.scatter(trunc_steps, trunc_rewards, c="blue", s=20, alpha=0.5, label=f"truncation ({len(trunc_steps)})")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Step")
ax.set_ylabel("Reward at terminal/truncation")
ax.set_title("Terminal events over 200 steps")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Goal detection: high reward events

# %%
N_STEPS = 512
goal_events = []

for t in range(N_STEPS):
    prev_obs = obs.copy()
    actions = random_actions(env)
    obs, rew, term, trunc, info = env.step(actions)
    for i in range(env.num_agents):
        if rew[i] >= 0.5:
            target_start = env.ego_features + env.num_reward_coefs
            goal_dist = np.sqrt(prev_obs[i, target_start] ** 2 + prev_obs[i, target_start + 1] ** 2)
            goal_events.append((t, i, rew[i], goal_dist))

print(f"Goal-like events (reward >= 0.5): {len(goal_events)}")
if goal_events:
    events = np.array(goal_events)
    print(f"Reward range: [{events[:, 2].min():.3f}, {events[:, 2].max():.3f}]")
    print(f"Goal distance at event: mean={events[:, 3].mean():.3f}, std={events[:, 3].std():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(events[:, 2], bins=20, edgecolor="black", alpha=0.7, color="gold")
    axes[0].set_title("Reward magnitude at goal events")
    axes[0].set_xlabel("Reward")
    axes[1].scatter(events[:, 3], events[:, 2], alpha=0.5)
    axes[1].set_xlabel("Goal distance before event")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Goal distance vs reward")
    plt.tight_layout()
    plt.show()
else:
    print("No goal events detected in 512 steps with random actions")

# %% [markdown]
# ## Reward scale for PPO

# %%
all_rewards = rewards_history.flatten()
episodic_returns = rewards_history.sum(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(all_rewards[all_rewards != 0], bins=50, edgecolor="black", alpha=0.7)
axes[0].set_title(f"Per-step reward distribution (non-zero, N={(all_rewards != 0).sum()})")
axes[0].set_xlabel("Reward")

axes[1].hist(episodic_returns, bins=20, edgecolor="black", alpha=0.7, color="purple")
axes[1].set_title(f"Episodic return (100 steps): mean={episodic_returns.mean():.3f}")
axes[1].set_xlabel("Return")
plt.tight_layout()
plt.show()

print(f"Reward magnitude range: [{all_rewards.min():.5f}, {all_rewards.max():.5f}]")
print(f"Mean episodic return: {episodic_returns.mean():.4f} +/- {episodic_returns.std():.4f}")
if abs(episodic_returns.mean()) > 10:
    print("WARNING: large episodic returns, consider scaling")
if episodic_returns.std() < 1e-6:
    print("WARNING: near-zero return variance")

# %% [markdown]
# ## Action-reward correlation

# %%
STEPS_PER_ACTION = 20
action_rewards = {}

for a in range(env.single_action_space.nvec[0]):
    rews = []
    for _ in range(STEPS_PER_ACTION):
        actions = np.full((env.num_agents, len(env.single_action_space.nvec)), a, dtype=np.int64)
        obs, rew, term, trunc, info = env.step(actions)
        rews.append(rew.mean())
    action_rewards[a] = np.mean(rews)

fig, ax = plt.subplots(figsize=(10, 5))
actions_list = sorted(action_rewards.keys())
means = [action_rewards[a] for a in actions_list]
colors = ["green" if m > 0 else "red" for m in means]
labels = [f"{a // 3}L,{a % 3}R" for a in actions_list]
ax.bar(range(len(actions_list)), means, tick_label=labels, color=colors, edgecolor="black")
ax.set_xlabel("Action (longitudinal, lateral)")
ax.set_ylabel("Mean reward")
ax.set_title(f"Mean reward per action over {STEPS_PER_ACTION} steps")
ax.axhline(0, color="black", lw=0.5)
plt.tight_layout()
plt.show()
