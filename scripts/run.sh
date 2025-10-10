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
#SBATCH --array=0-23


echo "=== SLURM job $SLURM_JOB_ID on node $SLURM_NODELIST ==="

# Define parameter arrays
ENTROPY_LB=(0 0 0 0)
ENTROPY_UB=(0.001 0.01 0.1 0.5)
DISCOUNT_LB=(0.8 0.9 0.95)
DISCOUNT_UB=(1 1 1)
ORACLE_MODES=("True" "False")
SEED=69

# Calculate which combination to use based on array task ID
# 24 jobs = 4 entropy × 3 discount × 2 oracle
ORACLE_IDX=$((SLURM_ARRAY_TASK_ID / 12))
TEMP=$((SLURM_ARRAY_TASK_ID % 12))
ENTROPY_IDX=$((TEMP / 3))
DISCOUNT_IDX=$((TEMP % 3))

ENT_LB=${ENTROPY_LB[$ENTROPY_IDX]}
ENT_UB=${ENTROPY_UB[$ENTROPY_IDX]}
DIS_LB=${DISCOUNT_LB[$DISCOUNT_IDX]}
DIS_UB=${DISCOUNT_UB[$DISCOUNT_IDX]}
ORC=${ORACLE_MODES[$ORACLE_IDX]}

echo "=== Task $SLURM_ARRAY_TASK_ID: entropy=$ENT_LB-$ENT_UB, discount=$DIS_LB-$DIS_UB, oracle=$ORC ==="

# Run inside singularity container
singularity exec --nv \
  --overlay "$OVERLAY_FILE:ro" \
  "$SINGULARITY_IMAGE" \
  bash -c "
    set -e

    source ~/.bashrc
    cd /scratch/mmk9418/projects/PufferDrive
    source .venv/bin/activate

    echo '=== Building extensions ==='
    python setup.py build_ext --inplace --force

    echo '=== Starting training ==='
    puffer train puffer_drive --wandb \
      --env.entropy-weight-lb $ENT_LB \
      --env.entropy-weight-ub $ENT_UB \
      --env.discount-weight-lb $DIS_LB \
      --env.discount-weight-ub $DIS_UB \
      --env.condition-type all \
      --env.oracle-mode $ORC \
      --train.seed $SEED
  "

echo "=== Job $SLURM_JOB_ID finished ==="
