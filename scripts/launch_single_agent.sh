#!/bin/bash
# Launch single-agent "speed run" training on the cluster via submit_cluster.py.
# This is the canonical nightly launch: code-isolated per run (no rebuild SIGBUS),
# container-wrapped, gpu-heartbeated, date-stamped wandb run names.
#
# Run on the login node (it sources the venv and submits from there):
#   ./scripts/launch_single_agent.sh
#
# Overridable via the environment:
#   PROGRAM_CONFIG  program_config YAML (default: single_agent_speed_run.yaml;
#                   e.g. single_agent_no_lane_vel.yaml for the reward ablation)
#   SEEDS           colon sweep passed to --args train.seed (default 0:1:2 -> 3 jobs)
#   ACCOUNT/PARTITION/TIME   SLURM overrides
#   PREFIX          run-name prefix (default <date>_single_agent)
#
# Examples:
#   SEEDS=0 PROGRAM_CONFIG=scripts/cluster_configs/single_agent_no_lane_vel.yaml \
#       ./scripts/launch_single_agent.sh
#   PARTITION=a100_tandon ./scripts/launch_single_agent.sh   # if h200 QOS is full
set -euo pipefail

PROGRAM_CONFIG="${PROGRAM_CONFIG:-scripts/cluster_configs/single_agent_speed_run.yaml}"
COMPUTE_CONFIG="${COMPUTE_CONFIG:-scripts/cluster_configs/nyu_greene.yaml}"
ACCOUNT="${ACCOUNT:-torch_pr_924_tandon_advanced}"
PARTITION="${PARTITION:-h200_tandon}"
TIME="${TIME:-720}"
SEEDS="${SEEDS:-0:1:2}"
PREFIX="${PREFIX:-$(date +%Y-%m-%d)_single_agent}"
DATE_STAMP="$(date +%Y-%m-%d)"

# Optional wandb rerouting: exporting WANDB_BASE_URL/WANDB_ENTITY in the
# launch environment pins the server/org for the jobs (e.g.
# WANDB_BASE_URL=https://api.wandb.ai WANDB_ENTITY=emerge_ for the public
# org instead of the cluster's self-hosted default).
EXTRA_WANDB_ARGS=()
[ -n "${WANDB_BASE_URL:-}" ] && EXTRA_WANDB_ARGS+=(--wandb-base-url "$WANDB_BASE_URL")
[ -n "${WANDB_ENTITY:-}" ] && EXTRA_WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")

source "/scratch/$USER/venvs/pufferdrive/bin/activate"

# Refresh the nightly trend runs
python scripts/nightly_report.py update || echo "trend update failed (non-fatal)"

# One submission per seed so we can pass a per-seed run_name (wandb display
# name like 2026-05-31_seed0)
IFS=':' read -ra SEED_LIST <<< "$SEEDS"
for SEED in "${SEED_LIST[@]}"; do
    python scripts/submit_cluster.py \
        --save_dir "/scratch/$USER/runs" \
        --prefix "$PREFIX" \
        --compute_config "$COMPUTE_CONFIG" \
        --program_config "$PROGRAM_CONFIG" \
        --container --heartbeat "${EXTRA_WANDB_ARGS[@]}" \
        --account "$ACCOUNT" --partition "$PARTITION" --time "$TIME" \
        --args "train.seed=$SEED" "run_name=${DATE_STAMP}_seed${SEED}" "wandb_group=${DATE_STAMP}"
done
