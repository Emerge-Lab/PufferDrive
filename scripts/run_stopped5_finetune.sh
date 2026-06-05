#!/bin/bash
# Fine-tune the zg6rezam multi-agent gigaflow policy with MAX_STOPPED_SECONDS=5.
#
# Modeled on scripts/run_nightly_best_local.sh (the local nightly_best launcher):
# same env/policy/train config, but FINE-TUNING from a checkpoint instead of
# training from scratch, for a fixed 20B aggregate env-steps, logging to its own
# wandb group/folder so it doesn't mix with the nightly runs.
#
# What's different from run_nightly_best_local.sh:
#   - --load-model-path <ckpt>  : load weights to fine-tune (policy/rnn arch +
#     obs KEYS_OF_INTEREST are inherited from the checkpoint's config.yaml).
#   - TOTAL_TIMESTEPS is a fixed 20B AGGREGATE (not per-rank * NUM_GPUS). pufferl
#     divides total by world size under DDP, so each rank does 20B/NUM_GPUS and
#     they sum back to 20B.
#   - wandb-group = max_stopped_5s_finetune (separate run/folder from the nightly).
#
# PREREQ: drive.h must have `#define MAX_STOPPED_SECONDS 5.0f` and the C ext
# rebuilt:  python setup.py build_ext --inplace --force
#
# Boolean flags must be Python literals (True/False), NOT yaml-style true/false,
# or the C binding rejects them. Eval rendering is disabled (the gigaflow eval
# still runs + logs metrics; only video frames are skipped) to keep an unattended
# run robust. Under DDP num_agents is PER-RANK; only rank 0 runs eval + wandb.
#
# Run from the repo root: ./scripts/run_stopped5_finetune.sh [NUM_GPUS] [NUM_AGENTS]
#   ./scripts/run_stopped5_finetune.sh          # 4 GPUs, 2048 agents/GPU
#   ./scripts/run_stopped5_finetune.sh 1 2048   # single-GPU smoke test
# Override the checkpoint via CKPT=..., the wandb run name via RUN_TAG=...
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from.
cd "$(dirname "$(readlink -f "$0")")/.."

NUM_GPUS="${1:-4}"
NUM_AGENTS="${2:-2048}"
# Fresh fine-tune: load only the WEIGHTS. We point at an alias dir that has the
# model + config.yaml but NO trainer_state.pt, so pufferl skips the optimizer/
# step-counter resume. (The real zg6rezam run finished at global_step=25B; a
# resume would set the counter to 25B and, with total_timesteps=20B, train zero
# steps.) This gives a fresh optimizer, fresh cosine LR schedule, and a fresh
# 20B step budget initialized from the checkpoint weights.
# To instead CONTINUE zg6rezam (resume optimizer/step), point CKPT at
# experiments/puffer_drive_zg6rezam/models/... and raise total_timesteps past 25B.
CKPT="${CKPT:-experiments/puffer_drive_zg6rezam_ftbase/models/model_puffer_drive_004769.pt}"
RUN_TAG="${RUN_TAG:-stopped5_ft_$(date +%Y-%m-%d_%H%M%S)}"

# Fixed 20B AGGREGATE env-steps. pufferl divides train.total_timesteps by world
# size under DDP, and the per-rank budgets sum back to this value, so pass 20B
# directly regardless of NUM_GPUS (4 GPUs -> 5B/rank -> 20B aggregate).
TOTAL_TIMESTEPS=20000000000

ARGS=(
  puffer_drive
  --env.simulation-mode gigaflow
  --env.map-dir pufferlib/resources/drive/binaries/carla_combined
  --env.num-maps 16
  --env.num-agents "$NUM_AGENTS"
  --env.min-agents-per-env 1
  --env.max-agents-per-env 150
  --env.use-map-cache 1
  --env.scenario-length 1200
  --env.resample-frequency 0
  --env.termination-mode 1
  --env.inactive-agent-threshold 0.4
  --env.dynamics-model jerk
  --env.target-type static
  --env.spawn-initial-speed 0.0
  --env.dt 0.3
  --env.traffic-light-behavior 1
  --env.collision-behavior 1
  --env.offroad-behavior 1
  --env.num-target-waypoints 3
  --env.min-waypoint-spacing 20.0
  --env.max-waypoint-spacing 60.0
  --env.goal-radius 2.0
  --env.goal-speed 3.0
  --env.obs-slots-lane-n 80
  --env.obs-slots-boundary-n 40
  --env.obs-slots-partners-n 16
  --env.obs-slots-traffic-controls-n 4
  --env.obs-range-partner-m 200.0
  --env.obs-range-road-front-m 200.0
  --env.obs-range-road-behind-m 40.0
  --env.obs-range-road-side-m 50.0
  --env.obs-range-traffic-control-m 100.0
  --env.obs-norm-xy-offset-m 200.0
  --env.obs-norm-goal-offset-m 200.0
  --env.obs-norm-road-seg-length-m 10.0
  --env.obs-norm-road-seg-width-m 5.0
  --env.obs-norm-veh-length-m 15.0
  --env.obs-norm-veh-width-m 10.0
  --env.obs-dropout-lane 0.5
  --env.obs-dropout-boundary 0.4
  --env.partner-blindness-prob 0.03
  --env.partner-blindness-trigger-prob 0.05
  --env.phantom-braking-prob 0.02
  --env.phantom-braking-trigger-prob 0.02
  --env.phantom-braking-duration 10
  --env.reward-conditioning True
  --env.reward-randomization True
  --env.reward-goal 1.0
  --env.reward-collision 1.5
  --env.reward-offroad 1.5
  --env.reward-stop-line 1.0
  --env.reward-comfort 0.05
  --env.reward-lane-align 0.025
  --env.reward-vel-align 1.0
  --env.reward-lane-center 0.005
  --env.reward-velocity 0.0025
  --env.reward-reverse 0.005
  --env.reward-timestep 2.5e-05
  --env.reward-overspeed 0.05
  --policy.input-size 256
  --policy.backbone-hidden-size 1024
  --policy.backbone-num-layers 3
  --policy.actor-hidden-size 1024
  --policy.actor-num-layers 0
  --policy.critic-hidden-size 1024
  --policy.critic-num-layers 0
  --policy.split-network True
  --policy.encoder-gigaflow True
  --policy.dropout 0.0
  --train.total-timesteps "$TOTAL_TIMESTEPS"
  --train.learning-rate 0.0005
  --train.minibatch-size 153600
  --train.max-minibatch-size 153600
  --train.update-epochs 3
  --train.bptt-horizon 128
  --train.compile True
  --train.precision bfloat16
  --train.normalize-rewards False
  --train.checkpoint-interval 500
  --train.optimizer adamw
  --train.seed 0
  # Load the checkpoint to fine-tune. policy/rnn arch + obs KEYS_OF_INTEREST come
  # from the checkpoint's sibling config.yaml; everything else from the args here.
  --load-model-path "$CKPT"
  --eval.validation-defaults.interval 250
  # validation_defaults hardcodes obs_slots_boundary_n=80; policy trains at 40, so
  # override the inline eval env too or it device-asserts at the first eval.
  --eval.validation-defaults.env.obs-slots-boundary-n 40
  --eval.validation-replay.enabled 0
  --eval.validation-gigaflow.render False
  --eval.behaviors-full-dir.enabled 0
  --eval.behaviors-hard-stop.enabled 0
  --eval.behaviors-highway-straight.enabled 0
  --eval.behaviors-lane-change.enabled 0
  --eval.behaviors-merge.enabled 0
  --eval.behaviors-parked-cars.enabled 0
  --eval.behaviors-roundabout.enabled 0
  --eval.behaviors-stopped-traffic.enabled 0
  --eval.behaviors-traffic-light-green.enabled 0
  --eval.behaviors-traffic-light-stop.enabled 0
  --eval.behaviors-unprotected-left.enabled 0
  --eval.behaviors-unprotected-right.enabled 0
  --wandb
  --wandb-project nightly-multi-agent
  --wandb-group max_stopped_5s_finetune
  --tag "$RUN_TAG"
)

# Args after NUM_GPUS/NUM_AGENTS pass straight through to override any config key
# (argparse takes the last value), e.g. `... 4 2048 --train.learning-rate 1e-4`.
if [ "$NUM_GPUS" -gt 1 ]; then
  exec torchrun --standalone --nnodes=1 --nproc-per-node="$NUM_GPUS" \
    -m pufferlib.pufferl train "${ARGS[@]}" "${@:3}"
else
  exec env CUDA_VISIBLE_DEVICES=0 puffer train "${ARGS[@]}" "${@:3}"
fi
