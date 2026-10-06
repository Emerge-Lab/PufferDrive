#!/bin/bash
# Train the SHIFT self-play base on one multi-GPU node with torchrun DDP
# (no SLURM, no container). Run from the repo root with the venv active.
#
#   ./scripts/launch_shift_pretrain_node.sh
#
# Overridable via the environment:
#   NUM_GPUS      ranks (default 4)
#   RUN_NAME      run id; rerunning with the same RUN_NAME and SAVE_DIR resumes
#                 from trainer_state.pt (default shift_selfplay_base)
#   SAVE_DIR      experiment root (default experiments/$RUN_NAME)
#   SEED          train.seed (default 0)
#   PROGRAM_CONFIG  per-rank overrides (default scripts/cluster_configs/shift/selfplay_pretrain.yaml)
#   EXTRA_ARGS    extra Hydra overrides, e.g. "wandb=false vec.num_workers=8"
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
RUN_NAME="${RUN_NAME:-shift_selfplay_base}"
SAVE_DIR="${SAVE_DIR:-experiments/$RUN_NAME}"
SEED="${SEED:-0}"
PROGRAM_CONFIG="${PROGRAM_CONFIG:-scripts/cluster_configs/shift/selfplay_pretrain.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Program-config keys are already Hydra override paths; emit key=value pairs.
CONFIG_ARGS=()
while IFS= read -r line; do
    CONFIG_ARGS+=("$line")
done < <(python - "$PROGRAM_CONFIG" <<'PY'
import sys, yaml
for key, value in yaml.safe_load(open(sys.argv[1])).items():
    print(f"{key}={value}")
PY
)

mkdir -p "$SAVE_DIR"
exec torchrun --standalone --nnodes=1 --nproc-per-node="$NUM_GPUS" \
    -m pufferlib.pufferl train puffer_drive \
    "${CONFIG_ARGS[@]}" \
    "train.seed=$SEED" "run_name=$RUN_NAME" "train.data_dir=$SAVE_DIR" \
    $EXTRA_ARGS
