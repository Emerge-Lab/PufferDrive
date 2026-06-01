#!/bin/bash
# Launch multi-agent "best launch" nightly training on the cluster via
# submit_cluster.py. Derived from oignons2 (emerge/temp_training); see
# scripts/cluster_configs/nightly_best_launch.yaml for the config.
# Three seeds per launch, date-stamped wandb run names.
#
# Run on the login node (sources the venv and submits from there):
#   ./scripts/launch_nightly_best.sh
#
# Overridable via the environment:
#   PROGRAM_CONFIG  default: scripts/cluster_configs/nightly_best_launch.yaml
#   SEEDS           colon sweep passed to --args train.seed (default 0:1:2)
#   ACCOUNT/PARTITION/TIME/MEM   SLURM overrides
#   PREFIX          run-name prefix (default <date>_multi_agent)
set -euo pipefail

PROGRAM_CONFIG="${PROGRAM_CONFIG:-scripts/cluster_configs/nightly_best_launch.yaml}"
COMPUTE_CONFIG="${COMPUTE_CONFIG:-scripts/cluster_configs/nyu_greene.yaml}"
ACCOUNT="${ACCOUNT:-torch_pr_924_tandon_advanced}"
PARTITION="${PARTITION:-h200_tandon}"
TIME="${TIME:-1800}"
MEM="${MEM:-192gb}"
SEEDS="${SEEDS:-0:1:2}"
PREFIX="${PREFIX:-$(date +%Y-%m-%d)_multi_agent}"
DATE_STAMP="$(date +%Y-%m-%d)"

source "/scratch/$USER/venvs/pufferdrive/bin/activate"

# One submission per seed so we can pass a per-seed run_name (wandb display
# name like 2026-06-01_seed0)
IFS=':' read -ra SEED_LIST <<< "$SEEDS"
for SEED in "${SEED_LIST[@]}"; do
    python scripts/submit_cluster.py \
        --save_dir "/scratch/$USER/runs" \
        --prefix "$PREFIX" \
        --compute_config "$COMPUTE_CONFIG" \
        --program_config "$PROGRAM_CONFIG" \
        --container --heartbeat \
        --account "$ACCOUNT" --partition "$PARTITION" --time "$TIME" --mem "$MEM" \
        --args "train.seed=$SEED" "run_name=${DATE_STAMP}_seed${SEED}"
done
