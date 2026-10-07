# SHIFT: progress log

Status of the `SHIFT` branch (off `3.0`) as of 2026-10-07. The branch implements
"Self-play is not enough: human imitation improves self-play autonomy in the
wild" (HIFT-PPO, called SHIFT here) plus the comparison arms of its Table A3,
including a proper HR-PPO baseline. Method details live in
[`hift_finetuning.md`](hift_finetuning.md); this page records what was built,
what was measured, and what is still open.

## 1. What the branch adds

### SHIFT fine-tuning (the paper's method)

| Ingredient | Knob |
|---|---|
| Quadratic similarity reward to the logged ego pose | `env.reward_expert_similarity` |
| Fixed-length sub-episodes inside a replay log | `env.episode_max_steps` |
| Random start offset per reset, drawn from the episode RNG | `env.init_step_jitter_steps` |
| Static-position linter: mask egos that barely move in the log | `env.static_expert_min_motion_m` |
| KL anchor to a frozen copy of the warm-start checkpoint | `train.kl_ref_coef` |

Agents-on-rails replay (`control_sdc_only`, other agents replaying their logs)
already existed. The SHIFT arm uses the paper's operating point: similarity
weight 1e-2, KL 2e-2, learning rate 1e-4, 6.6 s sub-episodes with 2.7 s start
offsets, which at the logs' 10 Hz is `dt=0.1`, 66 steps and 27 steps.

### Experiment matrix

One program config per arm in `scripts/cluster_configs/shift/`, launched with
`scripts/launch_shift_experiments.sh <arm>`.

| Arm | Config | What it isolates |
|---|---|---|
| Self-play base | `selfplay_pretrain.yaml` | the baseline and warm start |
| SHIFT | `shift.yaml` | the method |
| Extended self-play, matched data / matched cycles | `selfplay_extended.yaml` | "it's just more training" |
| BC only | `bc_only.yaml` | similarity reward alone (`env.expert_similarity_only`) |
| BC + RL | `bc_rl.yaml` | value of self-play pretraining |
| HR-PPO | `hrppo_anchor_bc.yaml`, then `hrppo.yaml` | the IL-anchored self-play pipeline |

### A proper HR-PPO baseline

Following "Human-like autonomy emerges from self-play and a pinch of human
data" (arXiv 2606.19370): self-play from scratch with sparse rewards, plus the
reverse KL `D_KL(pi_bc || pi)` at lambda 0.075 against a behavior-cloning policy
trained by supervised learning on logged actions.

- `train.kl_ref_direction=reference_to_policy` and `train.kl_ref_model_path`
  load and apply that anchor.
- `puffer bc puffer_drive` trains the anchor by cross-entropy on labels from the
  new `expert_tracking` SDC controller.
- The paper uses a delta-local action space; we deliberately keep the jerk
  bicycle model so every arm shares one sim and one evaluator.

### Making the jerk model fit human logs

The jerk model is stateful (longitudinal and lateral acceleration, steering),
and several fixes were needed before it could reproduce logged driving.

1. **State seeding at reset.** Replay resets zeroed the dynamics state; they now
   seed it from the log, removing a reset transient.
2. **Position-based speeds.** The bins' velocity field sits 0.16 m/s above the
   speed implied by logged positions; all seeding now derives from positions.
   Seeding from the velocity field made 48% of labels maximum braking.
3. **Open-loop labels.** `env.expert_tracking_teleport` (default on) snaps the
   ego back to the logged state before every label, so errors never accumulate.
4. **Rear-axle geometry.** The jerk model treated the box centre as the no-slip
   point. Logged centres slip sideways at yaw rate times 1.61 m (R^2 0.98),
   exactly the rear-axle lever arm. `env.jerk_rear_axle_slip` fixes it; it is
   on in every arm and defaults off elsewhere so old checkpoints stand.
5. **Glitch masking.** Labels are masked when the logged state lies outside the
   model's envelope (single-sample position glitches, rare extreme manoeuvres).
6. **Bounds are not the problem.** Widening the steering and acceleration
   limits did not change closed-loop failures; human kinematics already sit
   inside them at the 99.9th percentile.

`scripts/check_expert_tracking.py` measures the fit on any log set.

Tracking fidelity on a 303-log spread of `nuplan_mini_train`, 22-step
sub-episodes, rear-axle slip on:

| Mode | ADE p95 | peak error p95 | sub-episodes peaking over 0.5 m |
|---|---|---|---|
| Teleport (BC labels) | 0.7 cm | 2.0 cm | 0% |
| Closed loop | 5.1 cm | 12.6 cm | 2.0%, all log glitches |
| Zero-jerk baseline | 3.7 m | 11 m | – |

### nuPlan co-simulation port

`pufferlib/ocean/cosim/` (nuPlan side) was ported from `10_bernhard_dev` rather
than rebasing, so the base checkpoint stays valid. The port adds the sim hooks
the planner needs: externally owned partner slots, `goal_source=external`,
state, size, light and goal setters, and a lane-association refresh. The CARLA
bridge stayed behind. The ported C setter tests pass; devkit-dependent Python
tests skip without `nuplan` installed.

### Other fixes found along the way

- An uninitialized-variable warning in the jerk dynamics (not a live bug).
- The replay arms had inherited `dt=0.3` against 10 Hz logs; now pinned to 0.1.
- The anchor config inherited a 2,560-step scenario against 200-step logs, so
  it masked most steps and never cycled logs; now 200 steps, resampled each episode.

## 2. Compute and runs

Training runs on a 4x RTX PRO 6000 Blackwell vast.ai node (`ssh -p 34197
root@ssh1.vast.ai`), bootstrapped by `scripts/setup_standalone_node.sh` and
launched by `scripts/launch_shift_pretrain_node.sh`. W&B logs to
`wandb.ai/emerge_/shift-experiments`.

**Self-play base: finished.** 100B steps in about 33 hours at about 1M
steps/s, rear-axle slip on. Run:
<https://wandb.ai/emerge_/shift-experiments/runs/shift_selfplay_base>.
Checkpoint at `/workspace/runs/shift_selfplay_base/final_model.pt` on the node,
with a copy off-node.

| `nuplan_single` (1,000 val logs) | Self-play base |
|---|---|
| ADE / FDE (m) | 9.1 / 64.5 |
| Collision / at-fault | 27% / 7% |
| Off-road | 2.3% |
| Progress ratio | 0.30 |
| Puffer score | 0.41 |

FDE is large by design: the benchmark sends the ego along a route, not the
human's path.

## 3. HR-PPO anchor results

Anchors train on `nuplan_mini_train` (3,637 logs); evaluation is closed loop on
`nuplan_mini_val`.

| Anchor | Labels | Held-out accuracy | Off-road | Collision | Mean speed |
|---|---|---|---|---|---|
| 30 min, one-hot | 18,985 | 33% | 62% | 34% | 0.9 m/s |
| 3 h, one-hot | 114,157 | 38% | 59% | 38% | 0.9 m/s |
| 3 h, soft labels, tau 2e-5 | 114,157 | 32% (argmin) | 58% | 38% | 0.9 m/s |
| 3 h, soft labels, tau 1e-4 | 114,157 | 31% (argmin) | 56% | 39% | 0.9 m/s |

The majority class is about 30%. Diagnosis:

- **The fit works.** The actor overfits 584 and 5,100 samples to 100% training
  accuracy, and there is no observation aliasing (no duplicate observations).
- **The labels are knife-edge.** The median gap between the best and
  second-best jerk bin is 2e-5 m^2 over the tracking horizon. Neighbouring bins
  produce nearly identical motion, so the argmin label is noise relative to
  what the policy can observe, and more data barely helps.
- **Soft labels** (`bc.label_smoothing_temperature`, default 0) train against a
  softmax over the tracker's per-action errors. Closed loop they change almost
  nothing: off-road drops from 59% to 56-58%, collisions and speed are flat.
  The anchors' failure as drivers is compounding error from tiny-data BC, not
  label noise alone.

As drivers, the one-hot anchors crawl and drift off-road. That is typical of
tiny-data behavior cloning, and in HR-PPO the anchor only needs to shape the
action distribution, not drive on its own.

## 4. Open items

1. **Replay `dt` versus the base's `dt`.** Replay steps one log frame per sim
   step (`log_dt` is read from the bins but unused), so the replay arms run at
   `dt=0.1` while the base trained at `dt=0.3`. The paper runs both stages at
   0.3 s. Options: add a log stride to replay, fine-tune at 0.1, or retrain the
   base at 0.1. Undecided; the remaining arms wait on it.
2. **nuPlan co-simulation environment: working.** `scripts/setup_cosim_env.sh`
   builds a Python 3.10 env (the devkit's pins rule out 3.11) with PufferDrive
   installed without dependencies in its own checkout. On the node the planner,
   devkit, CaRL and `run_simulation` import, and all 28 bridge tests pass.
3. **nuPlan raw data.** Co-simulation needs the nuPlan DB logs and GPKG maps,
   not the bins. They exist on Greene (`/scratch/ev2237/data/nuplan`) and in
   `s3://pufferdrive-data/raw-files/nuplan`. Both local AWS credentials have
   expired, and Greene is reachable only while the user's session is live.
4. **Remaining arms.** SHIFT and the two extended self-play arms can start from
   the base now; BC only and BC + RL can start any time; HR-PPO waits on the
   anchor decision.
5. **Checkpoint durability.** `/workspace` on the node is not a persistent
   volume; a recycle or destroy wipes it.

## 5. Commits on `SHIFT`

```
324d8b7a Add optional soft BC labels from the expert tracker's per-action errors
b7569708 Anchor BC: pin scenario_length to the 10 Hz log and resample every episode
95f0d042 Port the nuPlan co-simulation bridge onto SHIFT
f82ca490 Pin dt=0.1 in the replay arms and rescale sub-episodes to 10 Hz
f6a4e5bd Install build dependencies before the non-isolated editable build
e1c5d3ed Install cu128 torch before the editable build on standalone nodes
339e49cf Add a bootstrap script for standalone GPU nodes
3ae22958 Size the self-play base per DDP rank and add a torchrun node launcher
1db56cf8 Size the SHIFT self-play base at 100B steps
748fe01c Mask BC labels outside the model envelope; rear-axle slip across the matrix
aa21a3b4 Add rear-axle slip to the jerk model behind env.jerk_rear_axle_slip
1c0c67b5 Open-loop expert tracking labels and position-based state seeding
5eb9aff9 Seed jerk-model state from the log at replay resets; add tracking fit check
a07da488 Set up the SHIFT experiment matrix and a supervised HR-PPO anchor
6c09ee90 Fix uninitialized jerk locals warning in move_dynamics
456c632d Add human-imitation fine-tuning (HIFT-PPO) for self-play policies
```
