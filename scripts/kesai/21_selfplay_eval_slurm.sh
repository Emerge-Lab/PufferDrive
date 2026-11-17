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

# CARLA self-play eval of a trained run with the post-training eval's overrides, e.g.
#   sbatch --export=ALL,SCENARIO_LENGTH=6000,EVAL_OUTPUT_DIR_NAME=eval80k_400s,WANDB_REPORT=0 scripts/kesai/21_selfplay_eval_slurm.sh
# Overridable env (default):
#   RUN_NAME (k_scaled_0059_1000)  the experiments/ run whose final_model.pt is evaluated
#   EVAL_OUTPUT_DIR_NAME (eval80k) output folder in the run, so re-evals never overwrite the training run's eval
#   TOTAL_SCENARIOS (80000), SCENARIO_LENGTH (the benchmark's)
#   WANDB_REPORT (1)               km/infraction -> the run's final_eval/selfplay_km_per_infraction; 0 for A/B variants
echo "START TIME: $(date)"
start=$(date +%s)

export RUN_NAME=${RUN_NAME:-k_scaled_0059_1000}
export MODEL_PATH=$PD/experiments/${RUN_NAME}/final_model.pt
[ -f "${MODEL_PATH}" ] || { echo "missing checkpoint ${MODEL_PATH}" >&2; exit 1; }
EVAL_OUTPUT_DIR_NAME=${EVAL_OUTPUT_DIR_NAME:-eval80k}
TOTAL_SCENARIOS=${TOTAL_SCENARIOS:-80000}
echo "run ${RUN_NAME}  output dir ${EVAL_OUTPUT_DIR_NAME}  scenarios ${TOTAL_SCENARIOS}  scenario_length ${SCENARIO_LENGTH:-benchmark default}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

.venv/bin/python scripts/parallel_eval.py carla \
    --total-scenarios ${TOTAL_SCENARIOS} \
    --num-nodes ${SLURM_JOB_NUM_NODES} \
    --shards-per-node 8 \
    env.map_dir=$PD/opengigaflow/resources/carla \
    vec.num_envs=60 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    ${SCENARIO_LENGTH:+env.scenario_length=${SCENARIO_LENGTH}} \
    eval.output_dir_name=${EVAL_OUTPUT_DIR_NAME} \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=${WANDB_REPORT:-1}

end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start))"
