#!/bin/bash
#SBATCH --job-name diag_reset_loop
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 0-02:00
#SBATCH --gres gpu:8
#SBATCH --mem=1007G
#SBATCH --cpus-per-task 144
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/log_%a_%A.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/log_%a_%A.err
#SBATCH --partition dev
#SBATCH --array=0-1

# Short 1-node run to catch sub-envs that early-reset every tick right after
# creation (see environment/early_reset_short, environment/spawn_* and the
# [DRIVE DIAG] lines in the .err). Task 0 = training spawn noise, task 1 = noise off.
echo "START TIME: $(date)"
start=$(date +%s)

export SEED=1000
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    export RUN_NAME=diag_reset_noise_on_${SEED}
    SPAWN_NOISE_ARGS="env.spawn_lateral_offset_max_frac=1.0 env.spawn_heading_max_deg=30.0"
else
    export RUN_NAME=diag_reset_noise_off_${SEED}
    SPAWN_NOISE_ARGS="env.spawn_lateral_offset_max_frac=0.0 env.spawn_heading_max_deg=0.0"
fi
echo ${RUN_NAME}
export DATA_DIR=/home/bjaeger/PufferDrive/experiments/${RUN_NAME}
echo ${DATA_DIR}

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source .venv/bin/activate

# Only task 0 builds; concurrent in-place builds race on shared NFS build files.
BUILD_STATUS_FILE=/home/bjaeger/PufferDrive/experiments/logs/build_status_${SLURM_ARRAY_JOB_ID}
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    bash scripts/kesai/build_ext_if_changed.sh /home/bjaeger/PufferDrive
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

# 2B steps = ~15k ticks at 8 GPUs; the stuck state formed within ~1.5k ticks in the 64-GPU runs.
torchrun --standalone --nnodes=1 --nproc-per-node=8 --max_restarts=0 --start-method spawn \
    -m pufferlib.pufferl train puffer_drive \
    wandb=True \
    wandb_project=nightly-multi-long \
    wandb_group=diag_reset_loop \
    train.data_dir=${DATA_DIR} \
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla_128_affine_zones \
    env.num_maps=128 \
    train.name=${RUN_NAME} \
    run_name=${RUN_NAME} \
    train.total_timesteps=2000000000 \
    vec.num_envs=16 \
    train.compile=True \
    train.max_minibatch_size=131072 \
    train.minibatch_size=131072 \
    train.precision=bfloat16 \
    policy.fp32_heads=true \
    train.tf32=false \
    env.goal_speed_randomization=false \
    env.goal_reach_requires_speed=true \
    env.obs_partner_relative_velocity=true \
    env.pose_noise_xy_m=0.025 \
    env.pose_noise_yaw_deg=0.25 \
    env.speed_limit_random_prob=1.0 \
    env.conditioning_speed_scale=2.0 \
    policy.mask_padded_features=true \
    train.evaluation_benchmarks=null \
    train.seed=${SEED} \
    ${SPAWN_NOISE_ARGS} \
    tb=True

echo "Training exited with status $?"
echo "Summarize with: python scripts/kesai/summarize_reset_diag.py /home/bjaeger/PufferDrive/experiments/logs/log_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.err"
end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start))"
