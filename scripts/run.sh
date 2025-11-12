#!/bin/bash
#SBATCH --job-name=puffer_drive
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=128GB
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=pr_100_tandon_priority
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --array=0

# Define condition types array
CONDITION_TYPES=("none") # "reward" "entropy" "discount" "all"
CONDITION_TYPE=${CONDITION_TYPES[$SLURM_ARRAY_TASK_ID]}

singularity exec --nv \
 --overlay "$OVERLAY_FILE:ro" \
 "$SINGULARITY_IMAGE" \
 bash -c "
   set -e

   source ~/.bashrc
   cd /scratch/mmk9418/projects/Adaptive_Driving_Agent
   source .venv/bin/activate

   puffer train puffer_adaptive_drive --wandb --env.condition-type $CONDITION_TYPE --env.num-maps 100
 "


# puffer train puffer_adaptive_drive --wandb --env.condition-type $CONDITION_TYPE --env.num-maps 100
# puffer train puffer_drive --wandb --env.num-maps 1000
