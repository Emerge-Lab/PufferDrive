#!/bin/bash
#SBATCH --job-name=puffer_drive
#SBATCH --output=/scratch/mmk9418/logs/%A_%a_%x.out
#SBATCH --error=/scratch/mmk9418/logs/%A_%a_%x.err
#SBATCH --mem=128GB
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=pr_100_tandon_priority
#SBATCH --array=0-0

echo "=== SLURM job $SLURM_JOB_ID on node $SLURM_NODELIST ==="

# Run inside singularity container
singularity exec --nv \
  --overlay "$OVERLAY_FILE" \
  "$SINGULARITY_IMAGE" \
  bash -c "
    set -e

    source ~/.bashrc
    cd /scratch/mmk9418/projects/PufferDrive
    git checkout mohit/ada
    source .venv/bin/activate

    echo '=== Building extensions ==='
    python setup.py build_ext --inplace --force

    echo '=== Starting training ==='
    puffer train puffer_drive --wandb
  "

echo "=== Job $SLURM_JOB_ID finished ==="
