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

export RUN_NAME=k_scaled_0036_${SEED}
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
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla_128_affine \
    env.num_maps=128 \
    train.name=${RUN_NAME} \
    run_name=${RUN_NAME} \
    train.total_timesteps=1000000000000 \
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
    policy.mask_padded_features=true \
    train.evaluation_benchmarks=carla_fast \
    train.final_model_name=${FINAL_MODEL_NAME} \
    train.seed=${SEED} \
    tb=True

if [ ! -f ${MODEL_PATH} ]; then
    echo "Training did not produce ${MODEL_PATH}; skipping evaluation."
    exit 1
fi

# parallel_eval places one shard per allocated node via srun, so each shard's 64
# env workers get a full node's cores instead of sharing the batch host.
echo "Training done, evaluating ${MODEL_PATH}"
.venv/bin/python scripts/parallel_eval.py carla \
    --total-scenarios 40000 \
    --num-nodes 8 \
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla \
    vec.num_envs=64 \
    eval.reward_comfort=0.0 \
    eval.reward_lane_center=0.0075 \
    env.eval_perceived_size_margin_m=0.2 \
    eval.min_goal_spacing=20 \
    eval.max_goal_spacing=200 \
    env.disable_red_light_infractions=1 \
    env.traffic_light_junction_phases=0 \
    env.eval_standstill_jerk_deadband_mps3=1.5 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True

python -m pufferlib.pufferl eval puffer_drive nuplan_multi \
    env.map_dir=/home/shared/data/nuPlan/PufferDrive \
    vec.num_envs=64 \
    eval.num_agents=300 \
    eval.reward_comfort=0.0 \
    eval.reward_lane_center=0.0075 \
    env.eval_perceived_size_margin_m=0.0 \
    eval.disable_red_light_infractions=1 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True


end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
