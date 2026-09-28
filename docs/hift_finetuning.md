# Human-imitation fine-tuning (HIFT-PPO)

Post-train a self-play policy on logged human driving without changing the
training algorithm. The recipe follows "Self-play is not enough: human imitation
improves self-play autonomy in the wild": the pretrained self-play policy is
warm-started and fine-tuned with the same PPO + advantage-filtering loop, with
four additions.

| Ingredient | Knob | Paper value |
|---|---|---|
| Agents-on-rails log replay: the ego is policy-driven, every other agent replays its logged trajectory | `env.simulation_mode=replay env.control_mode=control_sdc_only env.non_sdc_controller=replay env.non_vehicle_controller=replay` | – |
| Short sub-episodes with a random start offset inside each log | `env.episode_max_steps`, `env.init_step_jitter_steps` | 22 steps (6.6 s at dt 0.3), offsets 0..9 |
| Static-position linter: mask sub-episodes whose logged ego moves less than a floor over the horizon | `env.static_expert_min_motion_m` | 1.0 m |
| Similarity reward `r = -w * ||(x, y)_ego - (x, y)_expert||^2` per step, on top of every self-play reward term | `env.reward_expert_similarity` | 1e-2 |
| Fixed reward-conditioning vector shared by every agent | `env.reward_conditioning=true env.reward_randomization=false` plus the per-term `env.reward_*` values | Table A4 of the paper |
| Forward KL penalty `beta * D_KL(pi_theta || pi_pre)` against a frozen copy of the loaded checkpoint | `train.kl_ref_coef` | 2e-2 |
| Lower learning rate | `train.learning_rate` | 1e-4 (5e-4 in pretraining) |

## How the pieces are wired

- **Similarity reward** (`drive.h`, `compute_rewards`): squared distance between the
  ego's simulated position and the logged pose at the current timestep. Steps with
  no valid logged pose contribute zero. It is logged as
  `reward_components/expert_similarity` and shown in the HTML replay viewer.
- **Sub-episodes** (`drive.h`, `c_step` and `c_reset`): `episode_max_steps` truncates
  an episode that many steps after its start step. On every replay-mode reset the
  start step is re-sampled uniformly in `[init_step, init_step + init_step_jitter_steps]`
  from the episode RNG, so the same seed reproduces the same offsets. Eval envs
  never jitter.
- **Static-position linter** (`drive.h`, `flag_static_expert`): at reset, an agent
  whose logged (x, y) displacement between the start step and the end of the
  horizon is below `static_expert_min_motion_m` is flagged for the episode and
  its transitions are masked out of the rollout buffer, like the erratic-agent
  perturbations.
- **KL anchor** (`pufferl.py`, `_ppo_loss`): when `train.kl_ref_coef > 0` the trainer
  deep-copies the loaded checkpoint before the first update, freezes it, and adds
  `kl_ref_coef * mean D_KL(pi_theta(.|s) || pi_pre(.|s))` over the minibatch to the
  PPO loss. The KL is exact: over the joint discrete action categorical, or the
  closed form for the Gaussian head. It is logged as `losses/reference_kl`. This
  needs `load_model_path` (or `load_id`) and is not supported with an RNN policy.

The value function, advantage filtering, action space, and network are unchanged.

## Running it

```bash
puffer train puffer_drive \
    load_model_path=experiments/<pretrained_run>/models/model_puffer_drive_XXXXXX.pt \
    env.simulation_mode=replay env.control_mode=control_sdc_only \
    env.non_sdc_controller=replay env.non_vehicle_controller=replay \
    env.goal_source=gt env.map_dir=<dir of replay .bin logs> env.num_maps=<count> \
    env.max_agents_per_env=1 env.num_agents=128 \
    env.episode_max_steps=22 env.init_step_jitter_steps=9 env.static_expert_min_motion_m=1.0 \
    env.reward_expert_similarity=0.01 env.reward_conditioning=true env.reward_randomization=false \
    train.kl_ref_coef=0.02 train.learning_rate=1e-4
```

`scripts/cluster_configs/hift_finetune.yaml` bundles these overrides for
`submit_cluster.py`. Delete the checkpoint's sibling `trainer_state.pt` first if
you want a fresh optimizer rather than a resumed one.

Notes:

- `episode_max_steps` and `init_step_jitter_steps` are in simulation steps; scale
  them to the sim `dt` and the replay log length (`scenario_length`) of your data.
- `reward_expert_similarity` has an irreducible floor from the dynamics mismatch
  between the logged vehicle and the jerk bicycle model, and an L2 target to a
  single log penalises equally valid alternatives. The paper sweeps show safety
  degrades above `1e-2` and human-likeness saturates above `1e-1`.
- With `kl_ref_coef` too small the fine-tune drifts into BC-only behaviour
  (comfortable but unsafe); too large and it stops learning. `2e-2` is the paper's
  operating point for `1500` cycles.
