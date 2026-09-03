#!/bin/bash
#SBATCH --job-name train_puffer
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 2-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=1007G
#SBATCH --cpus-per-task 144
#SBATCH --output /home/bjaeger/nightly_PufferDrive/experiments/logs/log_%a_%A.out
#SBATCH --error /home/bjaeger/nightly_PufferDrive/experiments/logs/log_%a_%A.err
#SBATCH --partition dev
#SBATCH --array=0-2

# print info about current job
echo "START TIME: $(date)"
start=$(date +%s)

# Seed each array task deterministically: 1000 + 1000 * array task id
SEED=$((1000 + 1000 * SLURM_ARRAY_TASK_ID))
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} -> train.seed=${SEED}"

export RUN_NAME=k_nightly_"$(date +%Y-%m-%d)"_${SEED}
echo ${RUN_NAME}

export DATA_DIR=/home/bjaeger/nightly_PufferDrive/experiments/${RUN_NAME}
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

# Only task 0 builds; concurrent in-place builds race on shared NFS build files.
BUILD_STATUS_FILE=/home/bjaeger/nightly_PufferDrive/experiments/logs/build_status_${SLURM_ARRAY_JOB_ID}
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    bash scripts/kesai/build_ext_if_changed.sh /home/bjaeger/nightly_PufferDrive
    BUILD_STATUS=$?
    echo ${BUILD_STATUS} > ${BUILD_STATUS_FILE}
else
    BUILD_WAIT_SECONDS=0
    BUILD_TIMEOUT_SECONDS=1800
    while [ ! -f ${BUILD_STATUS_FILE} ]; do
        if [ ${BUILD_WAIT_SECONDS} -ge ${BUILD_TIMEOUT_SECONDS} ]; then
            echo "Timed out after ${BUILD_TIMEOUT_SECONDS}s waiting for task 0 build; aborting."
            exit 1
        fi
        sleep 10
        BUILD_WAIT_SECONDS=$((BUILD_WAIT_SECONDS + 10))
    done
    BUILD_STATUS=$(cat ${BUILD_STATUS_FILE})
fi
if [ "${BUILD_STATUS}" -ne 0 ]; then
    echo "C extension build failed with status ${BUILD_STATUS}; aborting."
    exit 1
fi

torchrun --standalone --nnodes=1 --nproc-per-node=8 --max_restarts=0 --start-method spawn \
    -m pufferlib.pufferl train puffer_drive \
    wandb=True \
    wandb_project=nightly-multi-long \
    wandb_group="$(date +%Y-%m-%d)" \
    train.data_dir=${DATA_DIR} \
    env.map_dir=/home/bjaeger/nightly_PufferDrive/pufferlib/resources/drive/binaries/carla \
    train.name=${RUN_NAME} \
    run_name=${RUN_NAME} \
    train.total_timesteps=100000000000 \
    vec.num_envs=16 \
    train.compile=True \
    train.max_minibatch_size=131072 \
    train.minibatch_size=131072 \
    train.precision=bfloat16 \
    train.evaluation_benchmarks=carla_fast \
    train.final_model_name=${FINAL_MODEL_NAME} \
    train.seed=${SEED} \
    tb=True
if [ ! -f ${MODEL_PATH} ]; then
    echo "Training did not produce ${MODEL_PATH}; skipping evaluation."
    exit 1
fi

echo "Training done, evaluating ${MODEL_PATH}"
.venv/bin/puffer eval puffer_drive carla \
    vec.num_envs=16 \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True

.venv/bin/puffer eval puffer_drive nuplan_single \
    env.map_dir=/home/shared/data/nuPlan/PufferDrive \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True


end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
