#!/bin/bash
#SBATCH --job-name eval_nuplan
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=0
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --output experiments/logs/nuPlan_reactive_%j.out
#SBATCH --error experiments/logs/nuPlan_reactive_%j.err
#SBATCH --partition dev
# nuPlan Val14 eval on our cluster: our values for scripts/slurm/eval_nuplan.sh, which documents the settings.
# 0_train_8gpu and 1_train_multinode submit this job with RUN_DIR=<its run dir>; 4_eval_nuplan_nonreactive.sh with CHALLENGES.
source scripts/kesai/cluster_env.sh || exit 1

# drop inherited entries of other OpenGigaFlow checkouts, whose opengigaflow Python would import instead
export PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v "cosim_Puffer\|/PufferDrive" | paste -sd: -)

set -a  # every assignment up to "set +a" reaches scripts/slurm/eval_nuplan.sh
PY=$(conda info --base)/envs/carl_nuplan/bin/python
RUN_DIR=${RUN_DIR:-$PD/experiments/k_scaled_0059_1000}
CITY_BIN_DIR=${CITY_BIN_DIR:-$SHARED_DATA/nuplan/PufferDrive}
# shell env still exports the pre-rename nuPlan casing; pin the renamed paths here
NUPLAN_DATA_ROOT=$SHARED_DATA/nuplan
NUPLAN_MAPS_ROOT=$SHARED_DATA/nuplan/maps
NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-$HOME/nuplan_exp}
# CPU-bound; a worker plateaus at ~2 GB with every city map cached: 384 stay under ~770 GB of a kesai node's 1 TB
THREADS_PER_NODE=${THREADS_PER_NODE:-$([ "$CLUSTER" = kesai ] && echo 384 || echo 128)}
WANDB_REPORT=${WANDB_REPORT:-1}  # 0 for ablations
[ "$CITY_BIN_DIR" != "$SHARED_DATA/nuplan/PufferDrive" ] && EVAL_TAG=_bins$(basename "$CITY_BIN_DIR")
set +a

exec bash scripts/slurm/eval_nuplan.sh
