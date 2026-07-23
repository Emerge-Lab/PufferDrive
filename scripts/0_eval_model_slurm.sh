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
.venv/bin/puffer eval puffer_drive --eval_simulation replay --render 1 --render-backend obs_html --load-model-path /lustre/scwpod02/client/kyutai/kesai/bernhard/PufferDrive/experiments/bernhard_train_base_wandb1_run-namek_004_longrun_300B_total-timesteps300000000000_max-minibatch-size131072_minibatch-size131072_num-envs2_91bb92d/puffer_drive_j671z22p/best_models/best_trainer_state_002981.pt

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
