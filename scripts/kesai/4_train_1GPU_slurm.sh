#!/bin/bash
#SBATCH --job-name train_puffer
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 2-00:00
#SBATCH --gres gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task 18
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/log_%a_%A.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/log_%a_%A.err
#SBATCH --partition dev
#SBATCH --array=0

# print info about current job
echo "START TIME: $(date)"
start=$(date +%s)

# Seed each array task deterministically: 1000 + 1000 * array task id
SEED=$((1000 + 1000 * SLURM_ARRAY_TASK_ID))
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} -> train.seed=${SEED}"

export RUN_NAME=k_fast_0001_${SEED}
echo ${RUN_NAME}

export DATA_DIR=/home/bjaeger/PufferDrive/experiments/${RUN_NAME}
echo ${DATA_DIR}

export FINAL_MODEL_NAME=final_model.pt
export MODEL_PATH=${DATA_DIR}/${FINAL_MODEL_NAME}
echo ${MODEL_PATH}

# TODO could try to tune these. 1 Is probably best since Puffer parallelizes across all cores.
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh || exit 1
torchrun --standalone --nnodes=1 --nproc-per-node=1 --max_restarts=0 --start-method spawn \
    -m pufferlib.pufferl train puffer_drive \
    wandb=True \
    wandb_project=nightly-multi-long \
    wandb_group=emerge_ \
    train.data_dir=${DATA_DIR} \
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla \
    train.name=${RUN_NAME} \
    run_name=${RUN_NAME} \
    train.total_timesteps=1000000000 \
    vec.num_envs=16 \
    train.compile=True \
    train.max_minibatch_size=131072 \
    train.minibatch_size=131072 \
    train.precision=bfloat16 \
    train.evaluation_benchmarks=carla_fast \
    train.final_model_name=${FINAL_MODEL_NAME} \
    train.seed=${SEED} \
    tb=True

# Only evaluate a run that actually finished, otherwise the eval jobs below would
# score a stale final_model.pt from an earlier attempt (or fail on a missing one).
TRAIN_STATUS=$?
if [ ${TRAIN_STATUS} -ne 0 ]; then
    echo "Training exited with status ${TRAIN_STATUS}; skipping evaluation."
    exit ${TRAIN_STATUS}
fi
if [ ! -f ${MODEL_PATH} ]; then
    echo "Training finished but ${MODEL_PATH} is missing; skipping evaluation."
    exit 1
fi

echo "Training done, evaluating ${MODEL_PATH}"
.venv/bin/puffer eval puffer_drive carla \
    vec.num_envs=16 \
    eval.action_selection=mean \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True

.venv/bin/puffer eval puffer_drive nuplan_single \
    env.map_dir=/home/shared/data/nuPlan/PufferDrive \
    eval.action_selection=mean \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True


end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
