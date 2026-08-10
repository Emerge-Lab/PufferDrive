#!/bin/bash
#SBATCH --job-name train_puffer
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 2-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=1007G
#SBATCH --cpus-per-task 144
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/log_%a_%A.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/log_%a_%A.err
#SBATCH --partition dev
#SBATCH --array=0-2

# print info about current job
echo "START TIME: $(date)"
start=$(date +%s)

# Seed each array task deterministically: 1000 + 1000 * array task id
SEED=$((1000 + 1000 * SLURM_ARRAY_TASK_ID))
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} -> train.seed=${SEED}"

export RUN_NAME=k_exp_0004_${SEED}
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
python setup.py build_ext --inplace --force
torchrun --standalone --nnodes=1 --nproc-per-node=8 --max_restarts=0 --start-method spawn \
    -m pufferlib.pufferl train puffer_drive \
    wandb=True \
    wandb_project=nightly-multi-long \
    wandb_group=emerge_ \
    train.data_dir=${DATA_DIR} \
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla \
    train.name=${RUN_NAME} \
    run_name=${RUN_NAME} \
    train.total_timesteps=100000000000 \
    vec.num_envs=16 \
    train.compile=True \
    train.max_minibatch_size=196608 \
    train.minibatch_size=196608 \
    train.precision=bfloat16 \
    env.num_agents=192 \
    train.min_batch_size=786432 \
    train.bptt_horizon=256 \
    policy.action_type=discrete \
    env.action_type=continuous \
    train.adv_filter_enabled=False \
    train.evaluation_benchmarks=carla_fast \
    train.final_model_name=${FINAL_MODEL_NAME} \
    train.seed=${SEED} \
    train.adam_eps=0.00001 \
    train.adam_weight_decay=0.0 \
    train.learning_rate=0.00025 \
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

# 786432 = 192 * 16 * 256
# 131072 = 524288 / 4

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
