#!/bin/bash
#SBATCH --job-name=train_puffer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=/lustre/scwpod02/client/kyutai/kesai/bernhard/carla_closed_loop/results/logs/eval_server_%a_%A.out
#SBATCH --error=/lustre/scwpod02/client/kyutai/kesai/bernhard/carla_closed_loop/results/logs/eval_server_%a_%A.err
#SBATCH --partition=kyutai

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

.venv/bin/puffer train puffer_drive --wandb --vec.num-envs 16

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"