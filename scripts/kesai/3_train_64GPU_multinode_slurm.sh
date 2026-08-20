#!/bin/bash
#SBATCH --job-name train_puffer
#SBATCH --nodes 8                        # Number of nodes requested
#SBATCH --ntasks-per-node 1              # Run 1 srun task per node (which fires up torchrun)
#SBATCH --gres gpu:8                     # GPUs per node
#SBATCH --cpus-per-task 144
#SBATCH --mem=1007G
#SBATCH --time 3-00:00
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/log_%a_%A.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/log_%a_%A.err
#SBATCH --partition dev

# Set up PyTorch Distributed Rendezvous parameters from Slurm variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

echo "START TIME: $(date)"
echo "Master node: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Total nodes: ${WORLD_SIZE}"
start=$(date +%s)

export SEED=1000

export RUN_NAME=k_scaled_0012_${SEED}
echo ${RUN_NAME}

export DATA_DIR=/home/bjaeger/PufferDrive/experiments/${RUN_NAME}
echo ${DATA_DIR}

# Name the training run writes its final model under, so the eval steps below can
# find it without knowing the last epoch number. Passed to train as
# train.final_model_name so the two can never disagree.
export FINAL_MODEL_NAME=final_model.pt
export MODEL_PATH=${DATA_DIR}/${FINAL_MODEL_NAME}
echo ${MODEL_PATH}

# Thread limit limits CPU thrashing across worker environments
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh /home/bjaeger/PufferDrive || exit 1
# Execute torchrun across all nodes using srun
srun torchrun \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --nproc-per-node=8 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --max_restarts=0 \
    --start-method spawn \
    -m pufferlib.pufferl train puffer_drive \
    wandb=True \
    wandb_project=nightly-multi-long \
    wandb_group=emerge_ \
    train.data_dir=${DATA_DIR} \
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla_hole_fixes \
    train.name=${RUN_NAME} \
    run_name=${RUN_NAME} \
    train.total_timesteps=1000000000000 \
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
# srun blocks until every node's task exits, so this is the whole run's status.
TRAIN_STATUS=$?
if [ ${TRAIN_STATUS} -ne 0 ]; then
    echo "Training exited with status ${TRAIN_STATUS}; skipping evaluation."
    exit ${TRAIN_STATUS}
fi
if [ ! -f ${MODEL_PATH} ]; then
    echo "Training finished but ${MODEL_PATH} is missing; skipping evaluation."
    exit 1
fi

# No srun: evaluation is a single-node job, run here on the batch host rather than
# once per allocated node.
echo "Training done, evaluating ${MODEL_PATH}"
.venv/bin/python scripts/parallel_eval.py carla \
    --total-scenarios 40000 \
    --num-gpus 8 \
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla_hole_fixes \
    vec.num_envs=64 \
    vec.num_workers=16 \
    eval.reward_comfort=0.0 \
    eval.reward_lane_center=0.0075 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True

.venv/bin/puffer eval puffer_drive nuplan_single \
    env.map_dir=/home/shared/data/nuPlan/PufferDrive \
    vec.num_envs=64 \
    eval.max_sdc_replay_workers=64 \
    eval.reward_comfort=0.0 \
    eval.reward_lane_center=0.0075 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True


end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
