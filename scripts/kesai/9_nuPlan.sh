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
export CKPT=$PD/weights/mimolette/models/model_puffer_drive_003815.pt
export CITY_BIN_DIR=/home/shared/data/nuPlan/PufferDrive
export PYTHONPATH=$PD:$CARL_DEVKIT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export COSIM_DEBUG_BEV=0
export CALLBACKS="[simulation_log_callback]"
export WORKER=ray_distributed
export THREADS_PER_NODE=96
export GROUP=/home/bjaeger/cosim_Puffer/experiments/nuplan_val14_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}

bash "$PD/pufferlib/ocean/cosim/nuplan/run_nuplan_planner.sh"
