#!/bin/bash
# Train the nuPlan self-play reward-conditioned recipe (displacement included) on Greene, one job per seed.
#
# Run on the login node (it sources the venv and submits from there):
#   ./scripts/launch_nuplan_selfplay_reward_conditioned.sh
# Preview without submitting:
#   DRY=1 ./scripts/launch_nuplan_selfplay_reward_conditioned.sh
#
# Overridable via the environment:
#   SEEDS                colon list of train.seed values (default 0)
#   ACCOUNT/PARTITION/TIME   SLURM overrides
#   MAP_DIR              env.map_dir override (default: the experiment's nuplan_train_moving)
#   SAVE_DIR             output root
#   PREFIX               job-name prefix and wandb tag (default <date>_selfplay_conditioned)
#   LAUNCH_ID            folded into run_name, which doubles as the wandb run id, so relaunches start new runs
#   RUN_TAG              run_name stem (default selfplay-conditioned)
#   EXTRA_ARGS           space-separated extra key=value overrides
set -euo pipefail

PROGRAM_CONFIG="scripts/cluster_configs/nuplan_selfplay_reward_conditioned.yaml"
COMPUTE_CONFIG="scripts/cluster_configs/nyu_greene.yaml"
ACCOUNT="${ACCOUNT:-torch_pr_355_general}"
PARTITION="${PARTITION:-l40s_public}"
# >48h lands in the gpu168 QOS; <=48h gets gpu48, which allows 0 GPUs per user on l40s_public
TIME="${TIME:-4320}"
SAVE_DIR="${SAVE_DIR:-/scratch/$USER/runs}"
WANDB_PROJECT="${WANDB_PROJECT:-pufferdrive}"
WANDB_GROUP="${WANDB_GROUP:-self-play}"
SEEDS="${SEEDS:-0}"
PREFIX="${PREFIX:-$(date +%Y-%m-%d)_selfplay_conditioned}"
LAUNCH_ID="${LAUNCH_ID:-$(date +%Y-%m-%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-selfplay-conditioned}"

DRY_FLAG=()
[ "${DRY:-0}" = "1" ] && DRY_FLAG=(--dry)
MAP_ARGS=()
[ -n "${MAP_DIR:-}" ] && MAP_ARGS=("env.map_dir=$MAP_DIR")
read -ra EXTRA_ARGS_LIST <<< "${EXTRA_ARGS:-}"

source "/scratch/$USER/venvs/pufferdrive/bin/activate"

IFS=':' read -ra SEED_LIST <<< "$SEEDS"
for SEED in "${SEED_LIST[@]}"; do
    python scripts/submit_cluster.py \
        --save_dir "$SAVE_DIR" \
        --prefix "$PREFIX" \
        --compute_config "$COMPUTE_CONFIG" \
        --program_config "$PROGRAM_CONFIG" \
        --container --heartbeat \
        --account "$ACCOUNT" --partition "$PARTITION" --time "$TIME" \
        --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" \
        "${DRY_FLAG[@]}" \
        --args "train.seed=$SEED" "run_name=${RUN_TAG}-${LAUNCH_ID}-seed${SEED}" \
               "${MAP_ARGS[@]}" "${EXTRA_ARGS_LIST[@]}"
done

echo "All ${#SEED_LIST[@]} jobs launched!"
