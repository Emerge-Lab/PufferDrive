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

`scripts/cluster_configs/shift/shift.yaml` bundles these overrides for
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

## Experiment matrix

One program config per arm lives in `scripts/cluster_configs/shift/`, and
`scripts/launch_shift_experiments.sh <arm>` submits it with the seed sweep.
All arms share the policy architecture, the discrete jerk action space, and
the PPO update; they differ only in the columns below (Table A3 of the paper).

| Arm | Config | Environment | Rewards | Warm start | KL anchor | What it isolates |
|---|---|---|---|---|---|---|
| Self-play | evaluate the pretrained checkpoint | – | – | – | – | the baseline |
| SHIFT | `shift.yaml` | log replay | RL + similarity | self-play | `0.02`, `D_KL(pi \|\| pi_pre)` | the method |
| Extra self-play, matched data | `selfplay_extended.yaml` + `train.total_timesteps` of SHIFT | self-play | RL | self-play | none | more RL on the same budget |
| Extra self-play, matched cycles | `selfplay_extended.yaml` + SHIFT epochs x steps per epoch | self-play | RL | self-play | none | more RL for the same cycle count |
| BC only | `bc_only.yaml` | log replay | similarity only (`expert_similarity_only`) | none | none | imitation without safety terms |
| BC+RL | `bc_rl.yaml` | log replay | RL + similarity | none | none | the value of self-play pretraining |
| HR-PPO | `hrppo_anchor_bc.yaml` then `hrppo.yaml` | self-play | sparse: goal, collision, off-road | none | `0.075`, `D_KL(pi_bc \|\| pi)` | the IL-anchored self-play pipeline |

Evaluate every arm on the same replay benchmark, which reports ADE and FDE next
to the safety metrics:

```bash
puffer eval puffer_drive nuplan_single load_model_path=<run>/models/model_puffer_drive_XXXXXX.pt
```

`losses/reference_kl` and `environment/reward_components/expert_similarity` in
W&B show whether the anchor and the similarity term are doing anything.

## HR-PPO: a proper supervised anchor

HR-PPO ("Human-like autonomy emerges from self-play and a pinch of human data")
anchors self-play to a behavior-cloning policy trained by supervised learning
on logged actions, with the reverse KL added to the PPO loss:

`L = L_PPO + lambda * E_o[D_KL(pi_bc(.|o) || pi_theta(.|o))]`, `lambda = 0.075`.

That is a different reference from the similarity-reward "BC only" arm, so the
pipeline has two steps.

1. **Anchor.** `puffer bc puffer_drive` with `env.sdc_controller=expert_tracking`.
   The tracker drives the SDC through the sim's own jerk dynamics: at each step
   it forward-simulates every discrete action, holds zero jerk for a short
   horizon, and keeps the action whose poses stay closest to the log. It writes
   that action back into the actions buffer as the label. On the bundled nuPlan
   log it stays under 1 cm ADE while a zero-jerk policy drifts by metres, and it
   uses the full spread of brake, coast, and accelerate bins. `puffer bc` then
   fits the actor by cross-entropy with early stopping on a held-out split and
   saves `models/model_puffer_drive_bc.pt` next to a `config.yaml`. Every
   infraction behaviour must be `ignore` so a brush with the log's collision
   margin does not freeze the tracked ego, and `dt` must match the log spacing.
2. **Self-play.** `hrppo.yaml`: sparse rewards, no reward conditioning, from
   scratch, `train.kl_ref_model_path` pointing at the anchor and
   `train.kl_ref_direction=reference_to_policy`. The policy block must match the
   anchor's so its weights load.

The one place the sim departs from the paper's sparse reward is the collision
penalty, which always adds a speed-scaled term on top of the coefficient.

### Action space and the stateful jerk model

The paper uses a discrete delta-local (Δx, Δy, Δψ) action space; this port
keeps the jerk bicycle model so every arm shares one sim and evaluator. That
model is stateful: longitudinal and lateral acceleration and the steering angle
carry over between steps. Replay-mode resets now seed that state from the log
at the start step (central-difference speed for longitudinal acceleration,
speed times logged yaw rate for lateral acceleration, the implied curvature for
steering), so a sub-episode starts mid-manoeuvre instead of from rest. The same
seeding applies to the policy-driven ego in the SHIFT arms, whose observation
includes those three quantities.

Check how well the jerk model fits your logs before trusting the anchor:

```bash
python scripts/check_expert_tracking.py --map-dir <replay .bin dir> --num-maps <n> --rounds 8
python scripts/check_expert_tracking.py --map-dir <replay .bin dir> --num-maps <n> --sdc-controller policy
```

The first prints the per-step tracking error over many random sub-episode
starts, the ADE/FDE distribution, and the label histogram; the second is the
zero-jerk baseline that shows the headroom. On the bundled nuPlan log, 22-step
sub-episodes at 10 Hz give a tracker ADE of 4 mm (p95 12 mm) and FDE of 5 mm,
with no sub-episode peaking above 3.2 cm, against 1.9 m ADE and 4.7 m FDE for
zero jerk. Before the state seeding the reset transient peaked at 15 cm (p95)
around step 7. A log set where the tracker's peak error regularly exceeds the
`--error-threshold-m` default of 0.5 m means the jerk bins or limits cannot
express that driving and the labels should not be trusted there.
