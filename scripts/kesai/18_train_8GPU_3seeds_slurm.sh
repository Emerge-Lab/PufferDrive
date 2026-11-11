#!/bin/bash
#SBATCH --job-name train_puffer
#SBATCH --nodes 1                        # Number of nodes requested
#SBATCH --ntasks-per-node 1              # Run 1 srun task per node (which fires up torchrun)
#SBATCH --gres gpu:8                     # GPUs per node
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --mem=0
#SBATCH --time 6-00:00
#SBATCH --output experiments/logs/train_3seeds_%a_%A.out
#SBATCH --error experiments/logs/train_3seeds_%a_%A.err
#SBATCH --partition dev
#SBATCH --array=0-1

source scripts/kesai/cluster_env.sh || exit 1

# Set up PyTorch Distributed Rendezvous parameters from Slurm variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

echo "START TIME: $(date)"
echo "Master node: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Total nodes: ${WORLD_SIZE}"
start=$(date +%s)

SEEDS=(1000 2000 1000 2000)
MINIBATCH_SIZES=(262144 262144 393216 393216)
RUN_PREFIXES=(k_scaled_0073 k_scaled_0073 k_scaled_0074 k_scaled_0074)

export SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MINIBATCH_SIZE=${MINIBATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
if [ -z "${SEED}" ]; then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} has no run configured; aborting."
    exit 1
fi
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} -> train.seed=${SEED} minibatch=${MINIBATCH_SIZE}"

export RUN_NAME=${RUN_PREFIXES[$SLURM_ARRAY_TASK_ID]}_${SEED}
echo ${RUN_NAME}

export DATA_DIR=$PD/experiments/${RUN_NAME}
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

# Hand-written CUDA slot-encoder kernels instead of the Triton ones (same math); triton to compare. Passed per command,
# never exported: sbatch copies this shell's environment into the eval jobs below, whose interpreters lack cuda-bindings.
SLOT_ENCODER_BACKEND=cuda
echo "slot encoder backend: ${SLOT_ENCODER_BACKEND}"

source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh $PD || exit 1
# 0073 / 0074 = the 0072 config (3_train_64GPU_multinode_slurm.sh) on one node per training, minibatch 262144 / 393216, seeds 1000 and 2000. Default 0073; 0074: sbatch --array=2-3 scripts/kesai/18_train_8GPU_3seeds_slurm.sh
# Execute torchrun across all nodes using srun
PUFFER_SLOT_ENCODER_BACKEND=${SLOT_ENCODER_BACKEND} srun --cpus-per-task="$SLURM_CPUS_PER_TASK" torchrun \
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
    env.map_dir=$PD/pufferlib/resources/drive/binaries/carla_128_affine_zones_stop_ndshoulders_v20 \
    env.num_maps=128 \
    env.obs_traffic_light_yellow_remaining=true \
    env.scripted_pedestrian_ratio=0.3 \
    train.name=${RUN_NAME} \
    run_name=${RUN_NAME} \
    train.total_timesteps=1000000000000 \
    vec.num_envs=128 \
    train.rollout_dtype=float16 \
    train.max_minibatch_size=${MINIBATCH_SIZE} \
    train.minibatch_size=${MINIBATCH_SIZE} \
    train.final_model_name=${FINAL_MODEL_NAME} \
    train.seed=${SEED} \
    tb=True

if [ ! -f ${MODEL_PATH} ]; then
    echo "Training did not produce ${MODEL_PATH}; skipping evaluation."
    exit 1
fi

# nuPlan (reactive and non-reactive), longest6 (v2 and v1) and AlpaSim evals need one 8-GPU node each: submit them as their own jobs before the long self-play eval below so they queue while it runs.
echo "Training done, submitting nuPlan reactive eval for ${MODEL_PATH}"
RUN_DIR=${DATA_DIR} sbatch scripts/kesai/9_nuPlan.sh \
    || echo "nuPlan eval submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/9_nuPlan.sh"
echo "Submitting nuPlan non-reactive eval for ${MODEL_PATH}"
RUN_DIR=${DATA_DIR} sbatch scripts/kesai/13_nuPlan_nonreactive.sh \
    || echo "nuPlan non-reactive eval submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/13_nuPlan_nonreactive.sh"
echo "Submitting longest6 v2 eval for ${MODEL_PATH}"
RUN_DIR=${DATA_DIR} sbatch scripts/kesai/11_carla_longest6_v2.sh \
    || echo "longest6 v2 eval submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/11_carla_longest6_v2.sh"
echo "Submitting longest6 v1 eval for ${MODEL_PATH}"
RUN_DIR=${DATA_DIR} sbatch scripts/kesai/14_carla_longest6_v1.sh \
    || echo "longest6 v1 eval submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/14_carla_longest6_v1.sh"
echo "Submitting AlpaSim eval for ${MODEL_PATH}"
RUN_DIR=${DATA_DIR} sbatch scripts/kesai/12_alpasim.sh \
    || echo "AlpaSim eval submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/12_alpasim.sh"

# parallel_eval runs 8 srun shards per node (one GPU and an eighth of the CPUs each, 60 blocking workers like
# training) so the eval uses every GPU and thread of the allocation; 60 workers x 5 scenarios = 300 scenarios per 9000-step episode.
echo "Follow-up evals submitted, evaluating ${MODEL_PATH}"
PUFFER_SLOT_ENCODER_BACKEND=${SLOT_ENCODER_BACKEND} .venv/bin/python scripts/parallel_eval.py carla \
    --total-scenarios 80000 \
    --num-nodes 1 \
    --shards-per-node 8 \
    env.map_dir=$PD/pufferlib/resources/drive/binaries/carla \
    vec.num_envs=60 \
    eval.reward_comfort=0.0 \
    eval.reward_lane_center=0.0075 \
    env.eval_perceived_size_margin_m=0.2 \
    eval.min_goal_spacing=20 \
    eval.max_goal_spacing=200 \
    env.disable_red_light_infractions=1 \
    env.disable_stop_sign_infractions=1 \
    env.traffic_light_junction_phases=0 \
    env.eval_standstill_jerk_deadband_mps3=1.5 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True

end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
