#!/bin/bash
# Plain single-GPU PufferDrive training job. Submit from the repo root:
#
#   mkdir -p slurm_logs   # slurmd opens the job log before the script runs
#   sbatch --account=<acct> --partition=<gpu-partition> scripts/train.sbatch
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
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/train_%j.log

set -euo pipefail

mkdir -p experiments

# NYU Greene: VENV_PATH=/scratch/$USER/venvs/pufferdrive
VENV_PATH="${VENV_PATH:-.venv}"

TRAIN_CMD=(
    puffer train puffer_drive
    --train.data-dir experiments
    --train.seed "${SEED:-0}"
    --train.total-timesteps 1_000_000_000
    # --wandb
    # --wandb-project pufferdrive
    # --wandb-group "$USER"
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
