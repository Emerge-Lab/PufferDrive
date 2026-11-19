#!/bin/bash
#SBATCH --job-name eval_longest6_v2
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=0
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --output experiments/logs/carla_longest6_v2_%j.out
#SBATCH --error experiments/logs/carla_longest6_v2_%j.err
#SBATCH --partition dev
# CARLA longest6 v2 eval on our cluster: our values for scripts/slurm/eval_longest6_v2.sh, which documents the settings
# (SEEDS, WORKERS, the COSIM_* ablation settings, ...). 0_train_8gpu and 1_train_multinode submit this job with
# RUN_DIR=<its run dir>.
source scripts/kesai/cluster_env.sh || exit 1

set -a  # every assignment up to "set +a" reaches scripts/slurm/eval_longest6_v2.sh
PY=${PY:-$HOME/miniconda3/envs/carl/bin/python}   # cp310: CARLA 0.9.15 ships no newer wheel
CARLA_ROOT=${CARLA_ROOT:-$HOME/CARLA_0.9.15}
CARL_WORK_DIR=${CARL_WORK_DIR:-$HOME/CaRL/CARLA}
RUN_DIR=${RUN_DIR:-$PD/experiments/k_scaled_0071_1000}
COSIM_TOWN_BIN_DIR=${COSIM_TOWN_BIN_DIR-$PD/opengigaflow/resources/carla_v21}  # the towns the run trained on; set but empty = the standard ones
WANDB_REPORT=${WANDB_REPORT:-1}  # 0 for ablations
set +a

exec bash scripts/slurm/eval_longest6_v2.sh
