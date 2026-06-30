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
# # 03 - Episode Metrics & Logging Debug
# Verify vec_log returns correct metrics, aggregation is sane, episode boundaries handled.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pufferlib.ocean.drive import binding
from notebooks.notebook_utils import make_drive_env, random_actions


env, obs, info = make_drive_env()

print(
    f"env ready: {env.num_agents} agents, obs={obs.shape}, act_shape={(env.num_agents, len(env.single_action_space.nvec))}"
)
print(
    f"ego_features={env.ego_features}, num_reward_coefs={env.num_reward_coefs}, obs_slots_partners_n={env.obs_slots_partners_n}, partner_features={env.partner_features}"
)
print(
    f"obs_slots_lane_kept={env.obs_slots_lane_kept}, obs_slots_boundary_kept={env.obs_slots_boundary_kept}, "
    f"lane_features={env.lane_features}, boundary_features={env.boundary_features}, stride={env.obs_lane_stride}/{env.obs_boundary_stride}"
)
print(
    f"obs_slots_traffic_controls_n={env.obs_slots_traffic_controls_n}, traffic_control_features={env.traffic_control_features}"
)

# %% [markdown]
# ## Single vec_log call

# %%
for _ in range(10):
    actions = random_actions(env)
    obs, rew, term, trunc, info = env.step(actions)

log = binding.vec_log(env.c_envs, env.num_agents)
print(f"vec_log type: {type(log)}")
if log:
    print(f"Keys: {sorted(log.keys())}")
    for k, v in sorted(log.items()):
        print(f"  {k}: {v}")
else:
    print("vec_log returned empty/None")

# %% [markdown]
# ## 512-step collection: all info dicts

# %%
N_STEPS = 512
all_logs = []
all_rewards = np.zeros((N_STEPS, env.num_agents))
all_terms = np.zeros((N_STEPS, env.num_agents))
all_truncs = np.zeros((N_STEPS, env.num_agents))

for t in range(N_STEPS):
    actions = random_actions(env)
    obs, rew, term, trunc, info = env.step(actions)
    all_rewards[t] = rew
    all_terms[t] = term
    all_truncs[t] = trunc
    if info:
        for log_entry in info:
            log_entry["_step"] = t
            all_logs.append(log_entry)

print(f"Collected {len(all_logs)} log entries over {N_STEPS} steps")
if all_logs:
    keys = set()
    for log in all_logs:
        keys.update(log.keys())
    keys.discard("_step")
    print(f"\n{'Metric':>25s} | {'count':>5s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
    print("-" * 75)
    for k in sorted(keys):
        vals = [log[k] for log in all_logs if k in log and isinstance(log[k], (int, float))]
        if vals:
            v = np.array(vals)
            print(f"{k:>25s} | {len(v):5d} {v.mean():10.4f} {v.std():10.4f} {v.min():10.4f} {v.max():10.4f}")

# %% [markdown]
# ## Metric definitions reference
#
# | Metric | Description |
# |--------|-------------|
# | score | Goals reached cleanly (no collision/offroad) |
# | collision_rate | Fraction of agents that collided |
# | offroad_rate | Fraction of agents that went off-road |
# | completion_rate | Fraction that reached goal (even with collision/offroad) |
# | lane_heading_aligned_rate | Fraction of steps with cos(theta) >= 0.5 (within ~60 deg of lane heading) |
# | lane_center_rate | Lane centering metric average (same as reward term) |
# | avg_collisions_per_agent | Average collision events per agent per episode |

# %% [markdown]
# ## Terminal / truncation timeline

# %%
term_per_step = all_terms.sum(axis=1)
trunc_per_step = all_truncs.sum(axis=1)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(term_per_step, label="terminals", alpha=0.7, color="red")
ax.plot(trunc_per_step, label="truncations", alpha=0.7, color="blue")
ax.set_xlabel("Step")
ax.set_ylabel("Count")
ax.set_title("Terminal/truncation events per step")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Total terminals: {all_terms.sum():.0f}, truncations: {all_truncs.sum():.0f}")
print(f"Terminals per step: mean={term_per_step.mean():.2f}, max={term_per_step.max():.0f}")

# %% [markdown]
# ## Agent lifecycle trajectories

# %%
TRACK_STEPS = 100
TRACK_AGENTS = min(5, env.num_agents)
xy_history = np.zeros((TRACK_STEPS, TRACK_AGENTS, 2))

for t in range(TRACK_STEPS):
    actions = random_actions(env)
    env.step(actions)
    states = env.get_global_agent_state()
    for i in range(TRACK_AGENTS):
        xy_history[t, i, 0] = states["x"][i]
        xy_history[t, i, 1] = states["y"][i]

fig, ax = plt.subplots(figsize=(10, 10))
for i in range(TRACK_AGENTS):
    ax.plot(xy_history[:, i, 0], xy_history[:, i, 1], "-o", markersize=2, alpha=0.7, label=f"agent {i}")
    ax.scatter(xy_history[0, i, 0], xy_history[0, i, 1], s=100, marker="s", zorder=10)
    ax.scatter(xy_history[-1, i, 0], xy_history[-1, i, 1], s=100, marker="*", zorder=10)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"{TRACK_AGENTS} agent trajectories over {TRACK_STEPS} steps")
ax.legend()
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Consistency checks

# %%
if all_logs:
    passed = 0
    failed = 0
    for log in all_logs:
        if "score" in log and "completion_rate" in log:
            if log["score"] > log["completion_rate"] + 1e-6:
                print(
                    f"FAIL: score ({log['score']:.4f}) > completion_rate ({log['completion_rate']:.4f}) at step {log['_step']}"
                )
                failed += 1
            else:
                passed += 1
        for rate_key in ["collision_rate", "offroad_rate", "completion_rate", "score"]:
            if rate_key in log:
                v = log[rate_key]
                if v < -1e-6 or v > 1.0 + 1e-6:
                    print(f"FAIL: {rate_key} = {v:.4f} outside [0,1] at step {log['_step']}")
                    failed += 1
                else:
                    passed += 1
    print(f"\nConsistency checks: {passed} passed, {failed} failed")
else:
    print("No logs to check")

# %% [markdown]
# ## Gigaflow agent dynamics

# %%
episode_lengths = []
agent_step_count = np.zeros(env.num_agents)
active_counts = []

for t in range(N_STEPS):
    active = (~np.all(all_rewards[: t + 1] == 0, axis=0) if t > 0 else np.ones(env.num_agents, dtype=bool)).sum()
    active_counts.append(active)
    for i in range(env.num_agents):
        agent_step_count[i] += 1
        if all_terms[t, i] or all_truncs[t, i]:
            episode_lengths.append(agent_step_count[i])
            agent_step_count[i] = 0

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(active_counts)
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Active agents")
axes[0].set_title("Active agent count over time")

if episode_lengths:
    axes[1].hist(episode_lengths, bins=30, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Episode length (steps)")
    axes[1].set_title(f"Episode length distribution (N={len(episode_lengths)})")
    print(f"Episode lengths: mean={np.mean(episode_lengths):.1f}, median={np.median(episode_lengths):.1f}")
else:
    print("No episodes completed")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Score vs cumulative reward

# %%
if all_logs and "score" in all_logs[0]:
    scores = [log["score"] for log in all_logs if "score" in log]
    log_steps = [log["_step"] for log in all_logs if "score" in log]
    cum_rew_at_log = [all_rewards[: t + 1].sum() / env.num_agents for t in log_steps]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cum_rew_at_log, scores, alpha=0.5)
    ax.set_xlabel("Avg cumulative reward up to step")
    ax.set_ylabel("Score")
    ax.set_title("Score vs cumulative reward")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No score data available")
