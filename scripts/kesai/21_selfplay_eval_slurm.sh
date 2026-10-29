#!/bin/bash
#SBATCH --job-name selfplay_eval
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 480              # whole kesai node without --exclusive (480 schedulable CPUs)
#SBATCH --gres gpu:8
#SBATCH --mem=0
#SBATCH --time 2:00:00
#SBATCH --output experiments/logs/selfplay_eval_%j.out
#SBATCH --error experiments/logs/selfplay_eval_%j.err
#SBATCH --partition dev

source scripts/kesai/cluster_env.sh || exit 1

# CARLA self-play eval only (no co-sim follow-ups), same overrides as the post-training eval in 3_train:
#   RUN_NAME=k_scaled_0059_1000 sbatch scripts/kesai/21_selfplay_eval_slurm.sh
#   RUN_NAME=k_scaled_0059_1000 SCENARIO_LENGTH=6000 EVAL_OUTPUT_DIR_NAME=eval80k_400s sbatch scripts/kesai/21_selfplay_eval_slurm.sh
# EVAL_OUTPUT_DIR_NAME names the output folder and the wandb prefix (final_<name>_carla), so re-evals never overwrite the training run's eval.
echo "START TIME: $(date)"
start=$(date +%s)

if [ -z "${RUN_NAME}" ]; then
    echo "RUN_NAME is required, e.g. RUN_NAME=k_scaled_0059_1000 sbatch $0" >&2
    exit 1
fi
export MODEL_PATH=$PD/experiments/${RUN_NAME}/final_model.pt
if [ ! -f "${MODEL_PATH}" ]; then
    echo "missing checkpoint ${MODEL_PATH}" >&2
    exit 1
fi
EVAL_OUTPUT_DIR_NAME=${EVAL_OUTPUT_DIR_NAME:-eval80k}
TOTAL_SCENARIOS=${TOTAL_SCENARIOS:-80000}
echo "run ${RUN_NAME}  output dir ${EVAL_OUTPUT_DIR_NAME}  scenarios ${TOTAL_SCENARIOS}  scenario_length ${SCENARIO_LENGTH:-benchmark default}"

export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

PUFFER_SLOT_ENCODER_BACKEND=${SLOT_ENCODER_BACKEND:-cuda} .venv/bin/python scripts/parallel_eval.py carla \
    --total-scenarios ${TOTAL_SCENARIOS} \
    --num-nodes ${SLURM_JOB_NUM_NODES} \
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
    ${SCENARIO_LENGTH:+env.scenario_length=${SCENARIO_LENGTH}} \
    eval.output_dir_name=${EVAL_OUTPUT_DIR_NAME} \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True

end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start))"
