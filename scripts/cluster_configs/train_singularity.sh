#!/bin/bash
# Usage: bash train_singularity.sh [extra pufferlib args...]
# All arguments are passed directly to pufferlib.pufferl train puffer_drive.
# Defaults are set below and can be overridden via CLI args.

set -e

# --- User config (change these per user) ---
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/ev2237}"
CODE_DIR="${SCRATCH_DIR}/code/PufferDrive"

# --- Environment setup ---
source /ext3/env.sh

# Redirect all caches to scratch (home has small quota)
export XDG_CACHE_HOME="${SCRATCH_DIR}/cache"
export WANDB_CACHE_DIR="${SCRATCH_DIR}/wandb_cache"
export WANDB_CONFIG_DIR="${SCRATCH_DIR}/wandb_config"
export WANDB_DATA_DIR="${SCRATCH_DIR}/wandb_data"
export WANDB_DIR="${SCRATCH_DIR}/wandb_data"
mkdir -p "$XDG_CACHE_HOME"

# --- Code isolation ---
# Create a symlink copy of the source tree with the .so copied (not linked),
# so that rebuilding the .so for another branch won't break running jobs.
# Parse --train.data-dir from args to determine where to put the isolated copy.
DATA_DIR=""
PREV_ARG=""
for arg in "$@"; do
    if [ "$PREV_ARG" = "--train.data-dir" ]; then
        DATA_DIR="$arg"
        break
    fi
    PREV_ARG="$arg"
done

if [ -n "$DATA_DIR" ]; then
    ISOLATED_CODE="${DATA_DIR}/code"
    mkdir -p "$DATA_DIR"
    cp -rs "$CODE_DIR" "$ISOLATED_CODE"
    # Replace .so symlinks with actual copies
    find "$ISOLATED_CODE" -name "*.so" -type l -exec sh -c 'cp --remove-destination "$(readlink -f "$1")" "$1"' _ {} \;
    cd "$ISOLATED_CODE"
else
    cd "$CODE_DIR"
fi

# --- Launch training ---
torchrun --standalone --nproc_per_node 1 -m pufferlib.pufferl train puffer_drive "$@"
