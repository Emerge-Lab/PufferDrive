#!/bin/bash
# Launch one arm of the SHIFT experiment matrix on the cluster via submit_cluster.py.
# See docs/hift_finetuning.md for what each arm isolates.
#
# Run on the login node:
#   ./scripts/launch_shift_experiments.sh <arm>
#
# Arms and what they need in the environment:
#   selfplay_pretrain        (nothing)                     the self-play base, rear-axle slip on
#   shift                    BASE_CKPT REPLAY_DIR NUM_MAPS
#   selfplay_matched_data    BASE_CKPT                      (same total_timesteps as shift)
#   selfplay_matched_cycles  BASE_CKPT                      (same epoch count as shift)
#   bc_only                  REPLAY_DIR NUM_MAPS
#   bc_rl                    REPLAY_DIR NUM_MAPS
#   hrppo_anchor             REPLAY_DIR NUM_MAPS            (single job, puffer bc)
#   hrppo                    BC_ANCHOR                      (models/model_puffer_drive_bc.pt from hrppo_anchor)
#
# Other overridable variables:
#   SEEDS      colon sweep for train.seed (default 0:1:2); ignored by hrppo_anchor
#   SHIFT_STEPS  total_timesteps of the shift arm used to size the self-play controls (default 150000000)
#   SHIFT_EPOCHS epoch count of the shift arm used for selfplay_matched_cycles (default 1500)
#   ACCOUNT / PARTITION / TIME / MEM  SLURM overrides
#   PREFIX     run-name prefix (default <date>_shift_<arm>)
#
# Example:
#   BASE_CKPT=/scratch/$USER/runs/selfplay/models/model_puffer_drive_010000.pt \
#   REPLAY_DIR=/scratch/$USER/data/nuplan_train NUM_MAPS=2000 ./scripts/launch_shift_experiments.sh shift
set -euo pipefail

ARM="${1:?usage: $0 <selfplay_pretrain|shift|selfplay_matched_data|selfplay_matched_cycles|bc_only|bc_rl|hrppo_anchor|hrppo>}"
CONFIG_DIR="scripts/cluster_configs/shift"
COMPUTE_CONFIG="${COMPUTE_CONFIG:-scripts/cluster_configs/nyu_greene.yaml}"
ACCOUNT="${ACCOUNT:-torch_pr_924_tandon_advanced}"
PARTITION="${PARTITION:-h200_tandon}"
TIME="${TIME:-1440}"
MEM="${MEM:-192gb}"
SEEDS="${SEEDS:-0:1:2}"
SHIFT_STEPS="${SHIFT_STEPS:-150000000}"
SHIFT_EPOCHS="${SHIFT_EPOCHS:-1500}"
DATE_STAMP="$(date +%Y-%m-%d)"
PREFIX="${PREFIX:-${DATE_STAMP}_shift_${ARM}}"

require() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "$ARM needs $name to be set" >&2
        exit 1
    fi
}

MAIN="-m pufferlib.pufferl train puffer_drive"
ARM_ARGS=()
case "$ARM" in
    selfplay_pretrain)
        PROGRAM_CONFIG="$CONFIG_DIR/selfplay_pretrain.yaml"
        ;;
    shift)
        require BASE_CKPT; require REPLAY_DIR; require NUM_MAPS
        PROGRAM_CONFIG="$CONFIG_DIR/shift.yaml"
        ARM_ARGS=("load_model_path=$BASE_CKPT" "env.map_dir=$REPLAY_DIR" "env.num_maps=$NUM_MAPS")
        ;;
    selfplay_matched_data)
        require BASE_CKPT
        PROGRAM_CONFIG="$CONFIG_DIR/selfplay_extended.yaml"
        ARM_ARGS=("load_model_path=$BASE_CKPT" "train.total_timesteps=$SHIFT_STEPS")
        ;;
    selfplay_matched_cycles)
        require BASE_CKPT
        PROGRAM_CONFIG="$CONFIG_DIR/selfplay_extended.yaml"
        # selfplay_extended.yaml: 40 envs x 3200 agents x 128 horizon transitions per epoch.
        STEPS_PER_EPOCH=$((40 * 3200 * 128))
        ARM_ARGS=("load_model_path=$BASE_CKPT" "train.total_timesteps=$((SHIFT_EPOCHS * STEPS_PER_EPOCH))")
        ;;
    bc_only|bc_rl)
        require REPLAY_DIR; require NUM_MAPS
        PROGRAM_CONFIG="$CONFIG_DIR/$ARM.yaml"
        ARM_ARGS=("env.map_dir=$REPLAY_DIR" "env.num_maps=$NUM_MAPS")
        ;;
    hrppo_anchor)
        require REPLAY_DIR; require NUM_MAPS
        PROGRAM_CONFIG="$CONFIG_DIR/hrppo_anchor_bc.yaml"
        MAIN="-m pufferlib.pufferl bc puffer_drive"
        ARM_ARGS=("env.map_dir=$REPLAY_DIR" "env.num_maps=$NUM_MAPS")
        SEEDS="0"
        ;;
    hrppo)
        require BC_ANCHOR
        PROGRAM_CONFIG="$CONFIG_DIR/hrppo.yaml"
        ARM_ARGS=("train.kl_ref_model_path=$BC_ANCHOR")
        ;;
    *)
        echo "unknown arm: $ARM" >&2
        exit 1
        ;;
esac

EXTRA_WANDB_ARGS=()
[ -n "${WANDB_BASE_URL:-}" ] && EXTRA_WANDB_ARGS+=(--wandb-base-url "$WANDB_BASE_URL")
[ -n "${WANDB_ENTITY:-}" ] && EXTRA_WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")

source "/scratch/$USER/venvs/pufferdrive/bin/activate"

IFS=':' read -ra SEED_LIST <<< "$SEEDS"
for SEED in "${SEED_LIST[@]}"; do
    python scripts/submit_cluster.py \
        --save_dir "/scratch/$USER/runs" \
        --prefix "$PREFIX" \
        --compute_config "$COMPUTE_CONFIG" \
        --program_config "$PROGRAM_CONFIG" \
        --main "$MAIN" \
        --container --heartbeat "${EXTRA_WANDB_ARGS[@]}" \
        --account "$ACCOUNT" --partition "$PARTITION" --time "$TIME" --mem "$MEM" \
        --args "train.seed=$SEED" "run_name=${PREFIX}_seed${SEED}" "wandb_group=${ARM}" "${ARM_ARGS[@]}"
done
