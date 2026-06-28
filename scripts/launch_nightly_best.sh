#!/bin/bash
# Launch multi-agent "nightly best" training on the cluster via submit_cluster.py.
# Mirrors launch_single_agent.sh but uses the oignons2-derived nightly_best.yaml
# (multi-agent gigaflow over 8 CARLA towns, 10B total steps). Code-isolated per
# run, container-wrapped, gpu-heartbeated, date-stamped wandb run names.
#
# Run on the login node (it sources the venv and submits from there):
#   ./scripts/launch_nightly_best.sh
#
# Overridable via the environment:
#   PROGRAM_CONFIG  program_config YAML (default: nightly_best.yaml)
#   SEEDS           colon sweep passed to --args train.seed (default 0:1:2 -> 3 jobs)
#   ACCOUNT/PARTITION/TIME   SLURM overrides
#   MEM             SLURM --mem (default 192gb; the multi-agent config plus
#                   inline validation_gigaflow eval can spike past 128gb at
#                   epoch 250)
#   PREFIX          run-name prefix (default <date>_multi_agent)
#
# Examples:
#   SEEDS=0 ./scripts/launch_nightly_best.sh                  # one-seed dry run
#   PARTITION=h100_tandon ./scripts/launch_nightly_best.sh    # if h200 QOS is full
set -euo pipefail

PROGRAM_CONFIG="${PROGRAM_CONFIG:-scripts/cluster_configs/nightly_best.yaml}"
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
# name like 2026-05-31_seed0).
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
