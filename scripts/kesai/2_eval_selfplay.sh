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
# Self-play eval of a trained run on our cluster: our values for scripts/slurm/eval_selfplay.sh, which documents the settings.
# TOTAL_SCENARIOS and SCENARIO_LENGTH empty = the benchmark's; WANDB_REPORT=1 writes km/infraction to the run's
# final_eval/selfplay_km_per_infraction, so keep it 0 for A/B variants such as other maps.
source scripts/kesai/cluster_env.sh || exit 1

set -a  # every assignment up to "set +a" reaches scripts/slurm/eval_selfplay.sh
RUN_DIR=$PD/experiments/k_scaled_0082_1000
MAP_DIR=$PD/opengigaflow/resources/carla_v21  # the 8 towns: the benchmark reads only the folder's first 8 sorted maps
EVAL_OUTPUT_DIR_NAME=eval_v21_maps
TOTAL_SCENARIOS=
SCENARIO_LENGTH=
WANDB_REPORT=0
WORKERS_PER_GPU=60  # 8 shards x 60 env workers = the node's 480 CPUs (x 5 scenarios = 300 per 9000-step episode)
set +a

exec bash scripts/slurm/eval_selfplay.sh
