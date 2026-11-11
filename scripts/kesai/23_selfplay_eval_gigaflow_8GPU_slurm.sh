#!/bin/bash
#SBATCH --job-name selfplay_eval_gigaflow
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 480              # whole kesai node without --exclusive (480 schedulable CPUs)
#SBATCH --gres gpu:8
#SBATCH --mem=0
#SBATCH --time 12:00:00                  # 80k scenarios on one node, ~4x the 4-node runtime of 21_selfplay_eval
#SBATCH --output experiments/logs/selfplay_eval_gigaflow_%j.out
#SBATCH --error experiments/logs/selfplay_eval_gigaflow_%j.err
#SBATCH --partition dev

source scripts/kesai/cluster_env.sh || exit 1

# CARLA self-play eval of one run on a single 8-GPU node: the post-training eval of 3_train with one setting changed,
# the lights of a junction cycle phase by phase. WANDB_REPORT=0: the run's wandb headline stays the standard eval's.
echo "START TIME: $(date)"
start=$(date +%s)

RUN_NAME=k_scaled_0076_1000
EVAL_OUTPUT_DIR_NAME=eval80k_junction_phases
TOTAL_SCENARIOS=80000
REWARD_COMFORT=0.05          # paper alpha_comfort
REWARD_LANE_CENTER=0.0038    # paper alpha_l-center
OBS_SLOTS_PARTNERS_N=40      # paper N_o: up to 40 closest agents; slots are max-pooled, any checkpoint transfers
TRAFFIC_LIGHT_JUNCTION_PHASES=1   # 3_train evaluates with 0, independent lights
WANDB_REPORT=0
SLOT_ENCODER_BACKEND=cuda

MODEL_PATH=$PD/experiments/${RUN_NAME}/final_model.pt
if [ ! -f "${MODEL_PATH}" ]; then
    echo "missing checkpoint ${MODEL_PATH}" >&2
    exit 1
fi
echo "run ${RUN_NAME}  output dir ${EVAL_OUTPUT_DIR_NAME}  scenarios ${TOTAL_SCENARIOS}"

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh $PD || exit 1

PUFFER_SLOT_ENCODER_BACKEND=${SLOT_ENCODER_BACKEND} .venv/bin/python scripts/parallel_eval.py carla \
    --total-scenarios ${TOTAL_SCENARIOS} \
    --num-nodes 1 \
    --shards-per-node 8 \
    env.map_dir=$PD/pufferlib/resources/drive/binaries/carla \
    vec.num_envs=60 \
    eval.reward_comfort=${REWARD_COMFORT} \
    eval.reward_lane_center=${REWARD_LANE_CENTER} \
    eval.obs_slots_partners_n=${OBS_SLOTS_PARTNERS_N} \
    env.eval_perceived_size_margin_m=0.2 \
    eval.min_goal_spacing=20 \
    eval.max_goal_spacing=200 \
    env.disable_red_light_infractions=1 \
    env.disable_stop_sign_infractions=1 \
    env.traffic_light_junction_phases=${TRAFFIC_LIGHT_JUNCTION_PHASES} \
    env.eval_standstill_jerk_deadband_mps3=1.5 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    eval.output_dir_name=${EVAL_OUTPUT_DIR_NAME} \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=${WANDB_REPORT}

end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start))"
