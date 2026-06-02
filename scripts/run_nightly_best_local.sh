#!/bin/bash
# Local (non-cluster) launch of the nightly_best_launch training.
# Direct `puffer train` equivalent of scripts/launch_nightly_best.sh +
# scripts/cluster_configs/nightly_best_launch.yaml, with no SLURM/submit_cluster.
#
# Activate your puffer env first (per CLAUDE.md) so `puffer` is on PATH.
# Boolean flags must be Python literals (True/False), NOT yaml-style true/false,
# or the C binding rejects them (e.g. "Failed to unpack keyword X as int").
# batch_size is auto = num_agents * bptt_horizon, so the on-GPU obs buffer scales
# with NUM_AGENTS; the 720000 default targets H200-class VRAM. Pass a smaller
# NUM_AGENTS (e.g. 2048) on smaller cards, or add --train.cpu-offload True.
#
# Run from the repo root: ./scripts/run_nightly_best_local.sh [GPU_INDEX] [NUM_AGENTS]
set -euo pipefail

GPU="${1:-0}"
NUM_AGENTS="${2:-720000}"

CUDA_VISIBLE_DEVICES="$GPU" puffer train puffer_drive \
  --env.simulation-mode gigaflow \
  --env.map-dir pufferlib/resources/drive/binaries/carla \
  --env.num-maps 8 \
  --env.num-agents "$NUM_AGENTS" \
  --env.min-agents-per-env 1 \
  --env.max-agents-per-env 150 \
  --env.use-map-cache 1 \
  --env.scenario-length 1200 \
  --env.resample-frequency 0 \
  --env.termination-mode 1 \
  --env.inactive-agent-threshold 0.4 \
  --env.dynamics-model jerk \
  --env.target-type static \
  --env.spawn-initial-speed 0.0 \
  --env.dt 0.3 \
  --env.traffic-light-behavior 1 \
  --env.collision-behavior 1 \
  --env.offroad-behavior 1 \
  --env.num-target-waypoints 3 \
  --env.min-waypoint-spacing 20.0 \
  --env.max-waypoint-spacing 60.0 \
  --env.goal-radius 2.0 \
  --env.goal-speed 3.0 \
  --env.obs-slots-lane-n 80 \
  --env.obs-slots-boundary-n 80 \
  --env.obs-slots-partners-n 16 \
  --env.obs-slots-traffic-controls-n 4 \
  --env.obs-range-partner-m 200.0 \
  --env.obs-range-road-front-m 200.0 \
  --env.obs-range-road-behind-m 40.0 \
  --env.obs-range-road-side-m 50.0 \
  --env.obs-range-traffic-control-m 100.0 \
  --env.obs-norm-xy-offset-m 200.0 \
  --env.obs-norm-goal-offset-m 200.0 \
  --env.obs-norm-road-seg-length-m 10.0 \
  --env.obs-norm-road-seg-width-m 5.0 \
  --env.obs-norm-veh-length-m 15.0 \
  --env.obs-norm-veh-width-m 10.0 \
  --env.obs-dropout-lane 0.5 \
  --env.obs-dropout-boundary 0.4 \
  --env.partner-blindness-prob 0.03 \
  --env.partner-blindness-trigger-prob 0.05 \
  --env.phantom-braking-prob 0.02 \
  --env.phantom-braking-trigger-prob 0.02 \
  --env.phantom-braking-duration 10 \
  --env.reward-conditioning True \
  --env.reward-randomization True \
  --env.reward-goal 1.0 \
  --env.reward-collision 1.5 \
  --env.reward-offroad 1.5 \
  --env.reward-stop-line 1.0 \
  --env.reward-comfort 0.05 \
  --env.reward-lane-align 0.025 \
  --env.reward-vel-align 1.0 \
  --env.reward-lane-center 0.005 \
  --env.reward-velocity 0.0025 \
  --env.reward-reverse 0.005 \
  --env.reward-timestep 2.5e-05 \
  --env.reward-overspeed 0.05 \
  --policy.input-size 256 \
  --policy.backbone-hidden-size 1024 \
  --policy.backbone-num-layers 3 \
  --policy.actor-hidden-size 1024 \
  --policy.actor-num-layers 0 \
  --policy.critic-hidden-size 1024 \
  --policy.critic-num-layers 0 \
  --policy.split-network True \
  --policy.encoder-gigaflow True \
  --policy.dropout 0.0 \
  --train.total-timesteps 10000000000 \
  --train.learning-rate 0.0005 \
  --train.minibatch-size 153600 \
  --train.max-minibatch-size 153600 \
  --train.update-epochs 3 \
  --train.bptt-horizon 128 \
  --train.compile True \
  --train.precision bfloat16 \
  --train.normalize-rewards False \
  --train.checkpoint-interval 500 \
  --train.optimizer adamw \
  --train.seed 0 \
  --eval.validation-defaults.interval 250 \
  --eval.validation-replay.enabled 0 \
  --eval.validation-gigaflow.render-backend egl \
  --eval.behaviors-full-dir.enabled 0 \
  --eval.behaviors-hard-stop.enabled 0 \
  --eval.behaviors-highway-straight.enabled 0 \
  --eval.behaviors-lane-change.enabled 0 \
  --eval.behaviors-merge.enabled 0 \
  --eval.behaviors-parked-cars.enabled 0 \
  --eval.behaviors-roundabout.enabled 0 \
  --eval.behaviors-stopped-traffic.enabled 0 \
  --eval.behaviors-traffic-light-green.enabled 0 \
  --eval.behaviors-traffic-light-stop.enabled 0 \
  --eval.behaviors-unprotected-left.enabled 0 \
  --eval.behaviors-unprotected-right.enabled 0
