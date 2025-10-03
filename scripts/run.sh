#!/bin/bash
#SBATCH --job-name=run
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --output=/n/netscratch/pehlevan_lab/Everyone/mkulkarni/pufferdrive/logs/%A_%a_%x.out
#SBATCH --error=/n/netscratch/pehlevan_lab/Everyone/mkulkarni/pufferdrive/logs/%A_%a_%x.err
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
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

# Define array of command arguments
COMMANDS=(
    # ""
    " --env.entropy-weight-ub 0.001"
    " --env.entropy-weight-ub 0.01"
    " --env.entropy-weight-ub 0.05"
    " --env.entropy-weight-ub 0.1"
    " --env.entropy-weight-ub 0.5"
    " --env.entropy-weight-ub 1"
    # "--env.oracle-mode False --env.condition-type none"
    # "--env.oracle-mode False --env.condition-type reward"
    # "--env.oracle-mode True --env.condition-type reward"
    # "--env.oracle-mode False --env.condition-type all"
    # "--env.oracle-mode True --env.condition-type all"
)

# Get the command args for this array task
ARGS="${COMMANDS[${SLURM_ARRAY_TASK_ID}]}"

# Run the command
puffer train puffer_drive --wandb ${ARGS}
