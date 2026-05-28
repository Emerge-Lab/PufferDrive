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

source "/scratch/$USER/venvs/pufferdrive/bin/activate"
python scripts/submit_cluster.py \
    --save_dir "/scratch/$USER/runs" \
    --prefix "$PREFIX" \
    --compute_config "$COMPUTE_CONFIG" \
    --program_config "$PROGRAM_CONFIG" \
    --container --heartbeat \
    --account "$ACCOUNT" --partition "$PARTITION" --time "$TIME" \
    --args "train.seed=$SEEDS"
