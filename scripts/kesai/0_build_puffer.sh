#!/bin/bash
#SBATCH --job-name=train_puffer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --output=experiments/logs/build_puffer_%j.out
#SBATCH --error=experiments/logs/build_puffer_%j.err
#SBATCH --partition=dev

source scripts/kesai/cluster_env.sh || exit 1

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

source .venv/bin/activate
python setup.py build_ext --inplace --force

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
