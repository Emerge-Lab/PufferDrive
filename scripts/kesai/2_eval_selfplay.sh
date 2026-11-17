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

# Self-play eval of a trained run's final_model.pt (the post-training eval's settings) into experiments/<run>/<EVAL_OUTPUT_DIR_NAME>.
# SCENARIO_LENGTH empty = the benchmark's; WANDB_REPORT=1 writes km/infraction to the run's final_eval/selfplay_km_per_infraction,
# so keep it 0 for A/B variants such as other maps.
RUN_NAME=k_scaled_0082_1000
MAP_DIR=$PD/opengigaflow/resources/carla_128_affine_aug_v21
EVAL_OUTPUT_DIR_NAME=eval_v21_maps
TOTAL_SCENARIOS=80000
SCENARIO_LENGTH=
WANDB_REPORT=0

echo "START TIME: $(date)"
start=$(date +%s)

export RUN_NAME
export MODEL_PATH=$PD/experiments/${RUN_NAME}/final_model.pt
[ -f "${MODEL_PATH}" ] || { echo "missing checkpoint ${MODEL_PATH}" >&2; exit 1; }
[ -d "${MAP_DIR}" ] || { echo "missing map dir ${MAP_DIR}" >&2; exit 1; }
echo "run ${RUN_NAME}  maps ${MAP_DIR}  output dir ${EVAL_OUTPUT_DIR_NAME}  scenarios ${TOTAL_SCENARIOS}  scenario_length ${SCENARIO_LENGTH:-benchmark default}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

.venv/bin/python scripts/parallel_eval.py selfplay \
    --total-scenarios ${TOTAL_SCENARIOS} \
    --num-nodes ${SLURM_JOB_NUM_NODES} \
    --shards-per-node 8 \
    env.map_dir=${MAP_DIR} \
    vec.num_envs=60 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    ${SCENARIO_LENGTH:+env.scenario_length=${SCENARIO_LENGTH}} \
    eval.output_dir_name=${EVAL_OUTPUT_DIR_NAME} \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=${WANDB_REPORT}

end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start))"
