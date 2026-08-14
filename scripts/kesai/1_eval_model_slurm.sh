#!/bin/bash
#SBATCH --job-name=eval_puffer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task=18
#SBATCH --output=/home/bjaeger/PufferDrive/experiments/logs/eval_%a_%A.out
#SBATCH --error=/home/bjaeger/PufferDrive/experiments/logs/eval_%a_%A.err
#SBATCH --partition=dev
#SBATCH --array=1-8

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

export RUN_NAME=k_nightly_2026-08-13_3000
MODEL_PATH=/home/bjaeger/PufferDrive/experiments/${RUN_NAME}/final_model.pt

source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh /home/bjaeger/PufferDrive || exit 1
.venv/bin/puffer eval puffer_drive carla \
    vec.num_envs=16 \
    eval.render_scenarios=true \
    eval.capture_observations=true \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True

.venv/bin/puffer eval puffer_drive nuplan_single \
    env.map_dir=/home/shared/data/nuPlan/PufferDrive \
    eval.render_scenarios=true \
    eval.capture_observations=true \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True


end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
