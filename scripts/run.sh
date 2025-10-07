#!/bin/bash
#SBATCH --job-name=rerun
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --output=/n/netscratch/pehlevan_lab/Everyone/mkulkarni/pufferdrive/logs/%A_%a_%x.out
#SBATCH --error=/n/netscratch/pehlevan_lab/Everyone/mkulkarni/pufferdrive/logs/%A_%a_%x.err
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_h100

module load cuda cudnn gcc/12.2.0-fasrc01
cd /n/home01/mkulkarni/projects/PufferDrive
source ~/.bashrc
source .venv/bin/activate
python setup.py build_ext --inplace --force

ENTROPY_VALUES=(0.0 0.0 0.0 0.0 0.001 0.001 0.01 0.01 0.01 0.1 0.1 0.1 0.5 0.5 0.5)
DISCOUNT_VALUES=(0.8 0.9 0.95 0.98 0.9 0.95 0.8 0.9 0.9 0.8 0.9 0.9 0.8 0.9 0.9)
ORACLE_MODES=(False False True False False True False False True False False True False False True)
SEEDS_MISSING=("42 69 420" "42 69 420" "42 69" "42 69 420" "42 69 420" "42 69 420" "42 69 420" "42 69 420" "42 69" "42 69 420" "42 69 420" "42 69" "42 69 420" "42 69 420" "42 69 420")

IDX=$SLURM_ARRAY_TASK_ID
ENT=${ENTROPY_VALUES[$IDX]}
DIS=${DISCOUNT_VALUES[$IDX]}
ORC=${ORACLE_MODES[$IDX]}
SEEDS=${SEEDS_MISSING[$IDX]}

echo "Rerunning entropy=$ENT discount=$DIS oracle=$ORC seeds=$SEEDS"

for S in $SEEDS; do
    echo "Running with seed $S..."
    puffer train puffer_drive --wandb \
        --env.entropy-weight-lb $ENT \
        --env.entropy-weight-ub $ENT \
        --env.discount-weight-lb $DIS \
        --env.discount-weight-ub $DIS \
        --env.condition-type all \
        --env.oracle-mode $ORC \
        --train.seed $S
done
