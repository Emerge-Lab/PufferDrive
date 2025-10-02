#!/bin/bash
#SBATCH --job-name=run
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --output=/n/netscratch/pehlevan_lab/Everyone/mkulkarni/pufferdrive/logs/%A_%a_%x.out
#SBATCH --error=/n/netscratch/pehlevan_lab/Everyone/mkulkarni/pufferdrive/logs/%A_%a_%x.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_h100

module load cuda
module load cudnn
module load gcc/12.2.0-fasrc01

echo "Running PufferDrive oracle mode sweep - Array task ${SLURM_ARRAY_TASK_ID}"

cd /n/home01/mkulkarni/projects/PufferDrive
source ~/.bashrc
source .venv/bin/activate
python setup.py build_ext --inplace --force

puffer train puffer_drive --wandb

# case ${SLURM_ARRAY_TASK_ID} in
#     0)
#         puffer train puffer_drive --wandb --env.oracle-mode False --env.condition-type "none"
#         ;;
#     1)
#         puffer train puffer_drive --wandb --env.oracle-mode False --env.condition-type "reward"
#         ;;
#     2)
#         puffer train puffer_drive --wandb --env.oracle-mode True --env.condition-type "reward"
#         ;;
#     3)
#         puffer train puffer_drive --wandb --env.oracle-mode False --env.condition-type "all"
#         ;;
#     4)
#         puffer train puffer_drive --wandb --env.oracle-mode True --env.condition-type "all"
#         ;;
# esac

