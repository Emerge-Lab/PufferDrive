#!/bin/bash
# Best-known-hyperparameter PufferDrive training job: multi-agent gigaflow
# over the 8 local CARLA maps, 10B steps. Hyperparameters mirror
# scripts/cluster_configs/nightly_best.yaml — keep in sync. Submit from the
# repo root:
#
#   mkdir -p slurm_logs   # slurmd opens the job log before the script runs
#   sbatch --account=<acct> --partition=<gpu-partition> scripts/best_train.sbatch
#
# Account and partition are cluster-specific, so pass them at submit time.
# Flags in TRAIN_CMD use the puffer CLI dash format (--train.total-timesteps),
# not the underscore format of scripts/cluster_configs/*.yaml; booleans are
# capitalized (True/False). The array copy-pastes directly to and from a
# console `puffer train puffer_drive ...` command.
#
# The job runs from the live checkout — no code isolation. Edits made while
# the job is queued change what runs. For sweeps, multi-GPU DDP, per-run code
# isolation, or NYU Greene's GPU heartbeat, use scripts/submit_cluster.py
# (long Greene runs need --heartbeat or the idle-GPU reclaimer kills them).
#SBATCH --job-name=pufferdrive-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192gb
#SBATCH --time=30:00:00
#SBATCH --output=slurm_logs/train_%j.log

set -euo pipefail

mkdir -p experiments

# NYU Greene: VENV_PATH=/scratch/$USER/venvs/pufferdrive
VENV_PATH="${VENV_PATH:-.venv}"

TRAIN_CMD=(
    puffer train puffer_drive
    --train.data-dir experiments
    --train.seed "${SEED:-0}"

    # Environment scale
    --env.simulation-mode gigaflow
    --env.map-dir pufferlib/resources/drive/binaries/carla
    --env.num-maps 8
    --env.num-agents 4096
    --env.min-agents-per-env 1
    --env.max-agents-per-env 120
    --env.use-map-cache 1
    --env.scenario-length 1280
    --env.resample-frequency 128000
    --env.termination-mode 1
    --env.inactive-agent-threshold 0.4
    --env.dynamics-model jerk
    --env.target-type static
    --env.dt 0.3

    # Goal setup — three sequential goals, route-based placement [20, 60m]
    --env.num-goals 3
    --env.min-goal-spacing 20.0
    --env.max-goal-spacing 60.0
    --env.goal-radius 2.0
    --env.goal-speed 1000.0

    # Observation shaping
    --env.obs-slots-lane-n 60
    --env.obs-slots-boundary-n 40
    --env.obs-slots-partners-n 12
    --env.obs-slots-traffic-controls-n 4
    --env.obs-range-partner-m 150.0
    --env.obs-range-road-front-m 150.0
    --env.obs-range-road-behind-m 40.0
    --env.obs-range-road-side-m 50.0
    --env.obs-range-traffic-control-m 150.0
    --env.obs-norm-xy-offset-m 150.0
    --env.obs-norm-goal-offset-m 200.0
    --env.obs-norm-road-seg-length-m 10.0
    --env.obs-norm-road-seg-width-m 5.0
    --env.obs-norm-veh-length-m 10.0
    --env.obs-norm-veh-width-m 5.0
    --env.obs-dropout-lane 0.2
    --env.obs-dropout-boundary 0.3

    # Perturbations (disabled in this config; eval's clean macro zeros them too)
    --env.partner-blindness-prob 0.0
    --env.partner-blindness-trigger-prob 0.0
    --env.phantom-braking-prob 0.0
    --env.phantom-braking-trigger-prob 0.0
    --env.phantom-braking-duration 10

    # Reward shaping
    --env.reward-conditioning False
    --env.reward-randomization False
    --env.reward-goal 0.5
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

    # Policy — 3x1024 backbone, split actor/critic
    --policy.ego-input-size 128
    --policy.partner-input-size 128
    --policy.lane-input-size 128
    --policy.boundary-input-size 128
    --policy.traffic-control-input-size 128
    --policy.context-input-size 64
    --policy.encoder-activation relu
    --policy.encoder-layer-norm True
    --policy.mask-padded-features False
    --policy.backbone-activation gelu
    --policy.backbone-hidden-size 1024
    --policy.backbone-num-layers 3
    --policy.backbone-layer-norm False
    --policy.actor-hidden-size 512
    --policy.actor-num-layers 0
    --policy.critic-hidden-size 512
    --policy.critic-num-layers 0
    --policy.shared-network False

    # Training — large minibatch, compiled bfloat16
    --train.total-timesteps 10_000_000_000
    --train.minibatch-size 128000
    --train.max-minibatch-size 128000
    --train.update-epochs 3
    --train.compile True
    --train.precision bfloat16
    --train.normalize-rewards False
    --train.checkpoint-interval 500

    # The yaml's eval.* overrides are not mirrored here; drive.ini's
    # [eval.*] defaults apply.

    # W&B logging
    # --wandb
    # --wandb-project nightly-multi
    # --wandb-group nightly-multi
)

# Launch record — the exact command and code version, greppable from the log.
echo "host:    $(hostname)"
echo "date:    $(date -Is)"
echo "commit:  $(git rev-parse HEAD || echo unknown)"
echo "command: ${TRAIN_CMD[*]}"

# NYU Greene needs the singularity wrap (modern glibc + CUDA live in the
# image); set both SINGULARITY_IMAGE and SINGULARITY_OVERLAY to enable it.
if [[ -n "${SINGULARITY_IMAGE:-}" && -n "${SINGULARITY_OVERLAY:-}" ]]; then
    # cd before sourcing so a relative VENV_PATH resolves against the repo
    # root even when the container does not preserve the working directory;
    # %q keeps every argument intact through the bash -c round-trip.
    INNER_CMD="cd $(printf '%q' "$PWD")"
    INNER_CMD+=" && source $(printf '%q' "$VENV_PATH/bin/activate")"
    INNER_CMD+=" && export PYTHONNOUSERSITE=1"
    INNER_CMD+=" && $(printf '%q ' "${TRAIN_CMD[@]}")"
    singularity exec --nv --overlay "${SINGULARITY_OVERLAY}:ro" "$SINGULARITY_IMAGE" \
        bash -c "$INNER_CMD"
else
    source "$VENV_PATH/bin/activate"
    "${TRAIN_CMD[@]}"
fi
