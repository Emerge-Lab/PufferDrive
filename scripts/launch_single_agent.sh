#!/bin/bash
# Single-agent nightly speed run (single_agent_speed_run.yaml) via submit_cluster.py: one code-isolated (no rebuild
# SIGBUS), container-wrapped, heartbeated job per seed with date-stamped wandb names. Run on the login node:
#   ./scripts/launch_single_agent.sh
# Overridable env (default):
#   PROGRAM_CONFIG (scripts/cluster_configs/single_agent_speed_run.yaml), COMPUTE_CONFIG (nyu_greene.yaml)
#   SEEDS (0:1:2)   one job per seed
#   ACCOUNT, PARTITION (h200_tandon; e.g. a100_tandon when its QOS is full), TIME (720 min)
#   PREFIX          run-name prefix (<date>_single_agent)
set -euo pipefail

PROGRAM_CONFIG="${PROGRAM_CONFIG:-scripts/cluster_configs/single_agent_speed_run.yaml}"
COMPUTE_CONFIG="${COMPUTE_CONFIG:-scripts/cluster_configs/nyu_greene.yaml}"
ACCOUNT="${ACCOUNT:-torch_pr_924_tandon_advanced}"
PARTITION="${PARTITION:-h200_tandon}"
TIME="${TIME:-720}"
SEEDS="${SEEDS:-0:1:2}"
PREFIX="${PREFIX:-$(date +%Y-%m-%d)_single_agent}"
DATE_STAMP="$(date +%Y-%m-%d)"

# exported WANDB_BASE_URL / WANDB_ENTITY pin the jobs' wandb server and org (e.g. https://api.wandb.ai, emerge_)
EXTRA_WANDB_ARGS=()
[ -n "${WANDB_BASE_URL:-}" ] && EXTRA_WANDB_ARGS+=(--wandb-base-url "$WANDB_BASE_URL")
[ -n "${WANDB_ENTITY:-}" ] && EXTRA_WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")

source "/scratch/$USER/venvs/pufferdrive/bin/activate"

python scripts/nightly_report.py update || echo "trend update failed (non-fatal)"

# one submission per seed for a per-seed run_name (2026-05-31_seed0)
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
