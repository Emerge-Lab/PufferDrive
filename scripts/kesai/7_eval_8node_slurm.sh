#!/bin/bash
#SBATCH --job-name eval_puffer_2node
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:8
#SBATCH --cpus-per-task 144
#SBATCH --mem=1007G
#SBATCH --time 1-00:00
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/eval_%a_%A.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/eval_%a_%A.err
#SBATCH --partition dev

# Eval-only twin of the post-training eval in 3_train_64GPU_multinode_slurm.sh: same CARLA
# self-play benchmark and follow-up co-sim jobs, on 2 nodes (16 GPUs) instead of 4.
echo "START TIME: $(date)"
start=$(date +%s)

export SEED=1000

export RUN_NAME=k_scaled_0045_${SEED}
echo ${RUN_NAME}

export DATA_DIR=/home/bjaeger/PufferDrive/experiments/${RUN_NAME}
echo ${DATA_DIR}

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

if [ ! -f ${MODEL_PATH} ]; then
    echo "${MODEL_PATH} is missing; nothing to evaluate."
    exit 1
fi

# parallel_eval places one shard per allocated node via srun, so each shard's 128
# env workers get a full node's cores instead of sharing the batch host.
echo "Evaluating ${MODEL_PATH}"
.venv/bin/python scripts/parallel_eval.py carla \
    --total-scenarios 40000 \
    --num-nodes 2 \
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla \
    vec.num_envs=128 \
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

# nuPlan (reactive and non-reactive), longest6 and AlpaSim evals need one 8-GPU node each: submit them as their own jobs so this 2-node allocation ends now.
echo "CARLA eval done, submitting nuPlan reactive eval for ${MODEL_PATH}"
RUN_DIR=${DATA_DIR} sbatch scripts/kesai/9_nuPlan.sh \
    || echo "nuPlan eval submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/9_nuPlan.sh"
echo "Submitting nuPlan non-reactive eval for ${MODEL_PATH}"
RUN_DIR=${DATA_DIR} sbatch scripts/kesai/13_nuPlan_nonreactive.sh \
    || echo "nuPlan non-reactive eval submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/13_nuPlan_nonreactive.sh"
echo "Submitting longest6 eval for ${MODEL_PATH}"
RUN_DIR=${DATA_DIR} sbatch scripts/kesai/11_carla_longest6.sh \
    || echo "longest6 eval submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/11_carla_longest6.sh"
echo "Submitting AlpaSim eval for ${MODEL_PATH}"
RUN_DIR=${DATA_DIR} sbatch scripts/kesai/12_alpasim.sh \
    || echo "AlpaSim eval submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/12_alpasim.sh"

end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
