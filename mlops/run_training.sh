#!/bin/bash
#
# Container entrypoint for PufferDrive training on GCP.

set -eo pipefail

echo "🚀 Starting PufferDrive training..."

# =============================================================================
# Argument Parsing
# =============================================================================

ACCELERATOR_COUNT=1
MODE="train"
TRAINING_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --accelerator-count)
            ACCELERATOR_COUNT="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            TRAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$ACCELERATOR_COUNT" =~ ^[0-9]+$ ]] || [[ "$ACCELERATOR_COUNT" -lt 1 ]]; then
    echo "❌ Error: --accelerator-count must be an integer >= 1 (got '$ACCELERATOR_COUNT')"
    exit 1
fi

# =============================================================================
# GPU Preflight Checks (fail fast, no CPU fallback)
# =============================================================================

echo "🔎 Validating GPU environment..."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "❌ Error: nvidia-smi not found. NVIDIA driver is missing or not visible in container."
    exit 1
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "❌ Error: nvidia-smi failed to query GPUs. Driver/runtime setup is invalid."
    exit 1
fi

VISIBLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed '/^$/d' | wc -l | tr -d ' ')
if [[ "$VISIBLE_GPUS" -lt "$ACCELERATOR_COUNT" ]]; then
    echo "❌ Error: requested ${ACCELERATOR_COUNT} GPU(s), but only ${VISIBLE_GPUS} visible via nvidia-smi."
    exit 1
fi

if ! EXPECTED_GPUS="$ACCELERATOR_COUNT" python - <<'PY'
import os
import sys
import torch

expected = int(os.environ["EXPECTED_GPUS"])

if torch.version.cuda is None:
    print("❌ Error: installed PyTorch is not CUDA-enabled (torch.version.cuda is None).")
    sys.exit(1)

if not torch.cuda.is_available():
    print("❌ Error: torch.cuda.is_available() is False. CUDA runtime/driver unavailable.")
    sys.exit(1)

count = torch.cuda.device_count()
if count < expected:
    print(f"❌ Error: PyTorch sees {count} CUDA device(s), but {expected} required.")
    sys.exit(1)

if expected > 1 and not torch.distributed.is_nccl_available():
    print("❌ Error: NCCL backend is unavailable for multi-GPU torchrun.")
    sys.exit(1)

if not torch.backends.cudnn.is_available():
    print("❌ Error: cuDNN is unavailable. CUDA stack is incomplete.")
    sys.exit(1)

print(f"✅ GPU preflight passed: torch CUDA={torch.version.cuda}, visible CUDA devices={count}.")
PY
then
    exit 1
fi

# =============================================================================
# Background Sync & Trap Setup
# =============================================================================

OUTPUT_DIR="/pufferdrive/training_output/"
mkdir -p "$OUTPUT_DIR"

SYNC_PID=""
GCS_DEST=""

if [[ -n "${CLOUD_PATH:-}" && "${NODE_RANK:-0}" == "0" ]]; then
    GCS_DEST="${CLOUD_PATH}/"
fi

do_sync() {
    # Use 'timeout' to guarantee gcloud never hangs the container infinitely
    if [[ "$MODE" == "sweep" ]]; then
        timeout 120s gcloud storage rsync -r "$OUTPUT_DIR/" "$GCS_DEST" --quiet || true
    else
        local run_dir
        run_dir=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
        if [[ -n "$run_dir" ]]; then
            timeout 120s gcloud storage rsync -r "$run_dir/" "$GCS_DEST" --quiet || true
        fi
    fi
}

cleanup() {
    local exit_code=$?
    # Clear traps instantly to prevent recursive loops
    trap - EXIT INT TERM ERR

    echo "🛑 Script exiting (code: $exit_code). Forcing cleanup..."

    if [[ -n "$SYNC_PID" ]]; then
        echo "🛑 Terminating background sync process (PID $SYNC_PID) and children..."
        # pkill -P kills the child 'sleep' or 'gcloud' holding standard output open
        pkill -TERM -P "$SYNC_PID" 2>/dev/null || true
        kill -9 "$SYNC_PID" 2>/dev/null || true
    fi

    if [[ -n "$GCS_DEST" ]]; then
        echo "☁️  Final GCS sync → $GCS_DEST (Max 2 mins)"
        do_sync
    fi

    echo "👋 Container exit complete."
    exit "$exit_code"
}

# Catch normal exits, set -e errors, AND system interrupts from GCP
trap cleanup EXIT INT TERM ERR

if [[ -n "$GCS_DEST" ]]; then
    sync_loop() {
        local sync_interval=90
        while true; do
            sleep "$sync_interval"
            do_sync
        done
    }
    sync_loop &
    SYNC_PID=$!
    echo "☁️  Background GCS sync started (pid=$SYNC_PID) → $GCS_DEST"
fi

# =============================================================================
# Training
# =============================================================================

if [[ "$MODE" == "sweep" ]]; then
    echo "Running hyperparameter sweep..."
    python -m pufferlib.pufferl sweep puffer_drive \
        --tb \
        --train.data-dir "$OUTPUT_DIR" \
        "${TRAINING_ARGS[@]}"
else
    torchrun \
        --standalone \
        --nproc-per-node="${ACCELERATOR_COUNT}" \
        --max-restarts=0 \
        -m pufferlib.pufferl train puffer_drive \
        --tb \
        --train.data-dir "$OUTPUT_DIR" \
        "${TRAINING_ARGS[@]}"
fi

echo "✅ Training completed successfully!"
