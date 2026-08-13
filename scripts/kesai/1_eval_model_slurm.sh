#!/bin/bash
#SBATCH --job-name=eval_puffer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task=18
#SBATCH --output=/home/bjaeger/PufferDrive/experiments/logs/eval_%a_%A.out
#SBATCH --error=/home/bjaeger/PufferDrive/experiments/logs/eval_%a_%A.err
#SBATCH --partition=dev
#SBATCH --array=1-8

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

SEED=$((1000 * SLURM_ARRAY_TASK_ID))
export RUN_NAME=k_exp_0009_${SEED}
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} -> ${RUN_NAME}"
MODEL_PATH=/home/bjaeger/PufferDrive/experiments/${RUN_NAME}/final_model.pt

source .venv/bin/activate

# Only task 1 builds; concurrent in-place builds race on shared NFS build files.
BUILD_STATUS_FILE=/home/bjaeger/PufferDrive/experiments/logs/build_status_${SLURM_ARRAY_JOB_ID}
if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    bash scripts/kesai/build_ext_if_changed.sh
    BUILD_STATUS=$?
    echo ${BUILD_STATUS} > ${BUILD_STATUS_FILE}
else
    BUILD_WAIT_SECONDS=0
    BUILD_TIMEOUT_SECONDS=1800
    while [ ! -f ${BUILD_STATUS_FILE} ]; do
        if [ ${BUILD_WAIT_SECONDS} -ge ${BUILD_TIMEOUT_SECONDS} ]; then
            echo "Timed out after ${BUILD_TIMEOUT_SECONDS}s waiting for task 1 build; aborting."
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
if [ ! -f ${MODEL_PATH} ]; then
    echo "${MODEL_PATH} is missing; skipping evaluation."
    exit 1
fi
.venv/bin/puffer eval puffer_drive carla \
    vec.num_envs=16 \
    eval.action_selection=mode \
    eval.output_dir_name=eval_mode \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True

.venv/bin/puffer eval puffer_drive nuplan_single \
    env.map_dir=/home/shared/data/nuPlan/PufferDrive \
    eval.action_selection=mode \
    eval.output_dir_name=eval_mode \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True


end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
