#!/bin/bash
#SBATCH --job-name=train_puffer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task=18
#SBATCH --output=/home/bjaeger/PufferDrive/experiments/k_scaled_0000/eval_nuPlan_%a_%A.out
#SBATCH --error=/home/bjaeger/PufferDrive/experiments/k_scaled_0000/eval_nuPlan_%a_%A.err
#SBATCH --partition=dev

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

source .venv/bin/activate
.venv/bin/puffer eval puffer_drive \
    --eval-simulation replay \
    --render 1 \
    --render-backend obs_html \
    eval.validation_replay.env.map_dir=/home/shared/data/nuPlan/PufferDrive \
    load_model_path=/home/bjaeger/PufferDrive/experiments/k_scaled_0000/puffer_drive_z09qq1nr/puffer_drive_z09qq1nr.pt \
    +render_results_dir=/home/bjaeger/PufferDrive/experiments/k_scaled_0000

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
