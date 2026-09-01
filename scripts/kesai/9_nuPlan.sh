#!/bin/bash
#SBATCH --job-name eval_nuplan
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --mem=1007G
#SBATCH --cpus-per-task 144
#SBATCH --output /home/bjaeger/cosim_Puffer/experiments/logs/log_%j.out
#SBATCH --error /home/bjaeger/cosim_Puffer/experiments/logs/log_%j.err
#SBATCH --partition dev
set -u

export PD=/home/bjaeger/cosim_Puffer
export PY=$(conda info --base)/envs/carl_nuplan/bin/python
RUN_DIR=/home/bjaeger/PufferDrive/experiments/k_scaled_0035_1000
# planner.py reads config.yaml from the checkpoint's run dir (parents[1]), so CKPT must be a models/*.pt
export CKPT=$(ls -t "$RUN_DIR"/models/model_*.pt 2>/dev/null | head -1)
: "${CKPT:?no model_*.pt found in $RUN_DIR/models}"
echo "Evaluating checkpoint: $CKPT"
export CITY_BIN_DIR=/home/shared/data/nuplan/PufferDrive
# shell env still exports the pre-rename nuPlan casing; pin the renamed paths here
export NUPLAN_DATA_ROOT=/home/shared/data/nuplan
export NUPLAN_MAPS_ROOT=/home/shared/data/nuplan/maps
export PYTHONPATH=$PD:$CARL_DEVKIT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export COSIM_DEBUG_BEV=0
export CALLBACKS="[simulation_log_callback]"
export WORKER=ray_distributed
export THREADS_PER_NODE=96
export GROUP=$PD/experiments/nuplan_val14_k_scaled_0035_1000_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}

bash "$PD/pufferlib/ocean/cosim/nuplan/run_nuplan_planner.sh"
