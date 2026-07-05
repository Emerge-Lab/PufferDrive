#!/bin/bash
# Enter the PufferDrive Singularity container. Run this after you already have
# a GPU allocation (e.g. via srun --pty bash). The venv, PYTHONPATH, and cache
# dirs are set up automatically so you can run puffer commands right away.
#
# Usage:
#   # Interactive shell (most common)
#   ./scripts/singularity_run.sh
#
#   # Run a one-shot command and exit
#   ./scripts/singularity_run.sh -- python eval_all.py --load-model-path ...
#
# Options:
#   --container-image P   Path to .sif image   (default: $IMAGE_PATH or system default)
#   --container-overlay P Path to .ext3 overlay (default: $OVERLAY_PATH or /scratch/$USER/...)

set -euo pipefail

SCRATCH_DIR="/scratch/${USER}"
IMAGE_PATH="${IMAGE_PATH:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
OVERLAY_PATH="${OVERLAY_PATH:-${SCRATCH_DIR}/images/PufferDrive/overlay-15GB-500K.ext3}"
VENV_PATH="${VENV_PATH:-${SCRATCH_DIR}/venvs/pufferdrive}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

COMMAND=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --container-image)   IMAGE_PATH="$2";   shift 2 ;;
        --container-overlay) OVERLAY_PATH="$2"; shift 2 ;;
        --)                  shift; COMMAND=("$@"); break ;;
        -h|--help)
            sed -n '2,20p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

PREAMBLE="source ${VENV_PATH}/bin/activate \
    && export PYTHONNOUSERSITE=1 \
    && export XDG_CACHE_HOME=${SCRATCH_DIR}/cache \
    && export WANDB_CACHE_DIR=${SCRATCH_DIR}/wandb_cache \
    && export WANDB_CONFIG_DIR=${SCRATCH_DIR}/wandb_config \
    && export WANDB_DATA_DIR=${SCRATCH_DIR}/wandb_data \
    && export WANDB_DIR=${SCRATCH_DIR}/wandb_data \
    && mkdir -p ${SCRATCH_DIR}/cache \
    && export PYTHONPATH=${PROJECT_ROOT}\${PYTHONPATH:+:\$PYTHONPATH} \
    && cd ${PROJECT_ROOT}"

if [[ ${#COMMAND[@]} -eq 0 ]]; then
    INNER_CMD="${PREAMBLE} && bash"
else
    USER_CMD_STR=$(printf '%q ' "${COMMAND[@]}")
    INNER_CMD="${PREAMBLE} && ${USER_CMD_STR}"
fi

SINGULARITY_CMD=(singularity exec --nv --overlay "${OVERLAY_PATH}:ro")

for cert_path in /etc/ssl/certs /etc/pki; do
    [[ -d "$cert_path" ]] && SINGULARITY_CMD+=(--bind "${cert_path}:${cert_path}:ro")
done

SINGULARITY_CMD+=("$IMAGE_PATH" bash -c "$INNER_CMD")

exec "${SINGULARITY_CMD[@]}"
