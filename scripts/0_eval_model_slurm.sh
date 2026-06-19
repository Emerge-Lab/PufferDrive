#!/bin/bash
#SBATCH --job-name=train_puffer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=28
#SBATCH --output=/lustre/scwpod02/client/kyutai/kesai/bernhard/carla_closed_loop/results/logs/eval_server_%a_%A.out
#SBATCH --error=/lustre/scwpod02/client/kyutai/kesai/bernhard/carla_closed_loop/results/logs/eval_server_%a_%A.err
#SBATCH --partition=kyutai

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

source .venv/bin/activate
.venv/bin/puffer eval puffer_drive --eval_simulation replay --render 1 --load-model-path /lustre/scwpod02/client/kyutai/kesai/bernhard/PufferDrive/experiments/puffer_eval/model.pt

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"