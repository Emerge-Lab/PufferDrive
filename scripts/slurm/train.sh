#!/bin/bash
#SBATCH --job-name train_opengigaflow
#SBATCH --nodes 1                        # GPUS_PER_NODE training ranks start on every node
#SBATCH --ntasks-per-node 1              # one srun task per node, which starts torchrun
#SBATCH --gres gpu:8
#SBATCH --exclusive                      # whole nodes: the env workers use every CPU
#SBATCH --mem=0
#SBATCH --time 2-00:00
#SBATCH --output experiments/logs/train_%j.out
#SBATCH --error experiments/logs/train_%j.err
##SBATCH --partition <partition>         # site-specific options: uncomment and set, or pass them to sbatch
# Trains one run with torchrun on the nodes of the allocation, then evaluates its final model. Submit from the checkout root,
# once experiments/logs exists; the arguments are Hydra overrides of opengigaflow/config/opengigaflow.yaml:
#   RUN_NAME=<run> sbatch scripts/slurm/train.sh train.seed=1 wandb=True wandb_project=<project>
# Settings, from the environment or edited here:
#   RUN_NAME                     the run lands in experiments/$RUN_NAME; submitting the same name again resumes it
#   GPUS_PER_NODE (8)            training ranks per node
#   SELFPLAY_EVAL (1)            1: scripts/slurm/eval_selfplay.sh runs in this allocation once training has ended
#   SELFPLAY_EVAL_MAP_DIR, SELFPLAY_EVAL_WANDB_REPORT (0)  that eval's MAP_DIR and WANDB_REPORT
#   FOLLOW_UP_EVALS              eval scripts submitted with RUN_DIR=<the run> once training has ended, space-separated
RUN_NAME=${RUN_NAME:-my_run}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SELFPLAY_EVAL=${SELFPLAY_EVAL:-1}
SELFPLAY_EVAL_MAP_DIR=${SELFPLAY_EVAL_MAP_DIR:-}
SELFPLAY_EVAL_WANDB_REPORT=${SELFPLAY_EVAL_WANDB_REPORT:-0}
FOLLOW_UP_EVALS=${FOLLOW_UP_EVALS:-}

source scripts/slurm/job_env.sh || exit 1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

printf 'START TIME: %s\nMaster node: %s\nTotal nodes: %s\n' "$(date)" "${MASTER_ADDR}:${MASTER_PORT}" "${WORLD_SIZE}"
start=$(date +%s)

RUN_DIR=$PD/experiments/$RUN_NAME
# fixed name: the evals below find the final model without knowing its epoch
FINAL_MODEL_NAME=final_model.pt
MODEL_PATH=$RUN_DIR/$FINAL_MODEL_NAME
echo "run $RUN_NAME -> $RUN_DIR"

# one BLAS/OpenMP thread per process: the env workers would otherwise thrash the CPUs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

source .venv/bin/activate
bash scripts/slurm/build_ext_if_changed.sh "$PD" || exit 1
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --max_restarts=0 \
    --start-method spawn \
    -m opengigaflow.rl train \
    train.data_dir="$RUN_DIR" \
    run_name="$RUN_NAME" \
    train.final_model_name="$FINAL_MODEL_NAME" \
    "$@"

[ -f "$MODEL_PATH" ] || { echo "Training did not produce $MODEL_PATH; skipping evaluation."; exit 1; }

# submitted before the long self-play eval below, so they queue while it runs
for eval_script in $FOLLOW_UP_EVALS; do
    echo "Submitting $eval_script for $MODEL_PATH"
    RUN_DIR=$RUN_DIR sbatch "$eval_script" || echo "submission failed; run by hand: RUN_DIR=$RUN_DIR sbatch $eval_script"
done

if [ "$SELFPLAY_EVAL" = "1" ]; then
    echo "Evaluating $MODEL_PATH"
    RUN_DIR=$RUN_DIR GPUS_PER_NODE=$GPUS_PER_NODE MAP_DIR=$SELFPLAY_EVAL_MAP_DIR WANDB_REPORT=$SELFPLAY_EVAL_WANDB_REPORT BUILD_EXT=0 \
        bash scripts/slurm/eval_selfplay.sh
fi

end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start))"
