#!/bin/bash
#SBATCH --job-name eval_longest6_v1
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=0
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --output experiments/logs/carla_longest6_v1_%j.out
#SBATCH --error experiments/logs/carla_longest6_v1_%j.err
#SBATCH --open-mode=append               # a requeued job keeps the log of the instance it resumes
#SBATCH --partition dev
# CARLA longest6 v1 eval on our cluster: our values for scripts/slurm/eval_longest6_v1.sh, which documents the settings
# (SEEDS, WORKERS, COMPLETE_RUN, the COSIM_* ablation settings, ...). 0_train_8gpu and 1_train_multinode submit this job
# with RUN_DIR=<its run dir>.
source scripts/kesai/cluster_env.sh || exit 1

set -a  # every assignment up to "set +a" reaches scripts/slurm/eval_longest6_v1.sh
GARAGE_PY=${GARAGE_PY:-$HOME/miniconda3/envs/garage/bin/python}   # cp37: the CARLA 0.9.10 egg
SERVER_PY=${SERVER_PY:-$HOME/miniconda3/envs/carl/bin/python}     # any OpenGigaFlow-capable python
CARLA_ROOT=${CARLA_ROOT:-$HOME/CARLA_0.9.10}
GARAGE_WORK_DIR=${GARAGE_WORK_DIR:-$HOME/carla_garage_1}
RUN_DIR=${RUN_DIR:-$PD/experiments/k_scaled_0071_1000}
COSIM_TOWN_BIN_DIR=${COSIM_TOWN_BIN_DIR-$PD/opengigaflow/resources/carla_v21}  # the towns the run trained on; set but empty = the standard ones
WANDB_REPORT=${WANDB_REPORT:-1}  # 0 for ablations
set +a

exec bash scripts/slurm/eval_longest6_v1.sh
