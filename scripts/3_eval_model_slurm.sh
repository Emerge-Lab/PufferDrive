#!/bin/bash
#SBATCH --job-name=eval_puffer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task=18
#SBATCH --output=/home/bjaeger/PufferDrive/experiments/k_scaled_0006/puffer_drive_nht7wfsc/eval2/eval_nuPlan_%a_%A.out
#SBATCH --error=/home/bjaeger/PufferDrive/experiments/k_scaled_0006/puffer_drive_nht7wfsc/eval2/eval_nuPlan_%a_%A.err
#SBATCH --partition=dev

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

RESULTS_PATH=/home/bjaeger/PufferDrive/experiments/k_scaled_0006/puffer_drive_nht7wfsc/eval2
MODEL_PATH=/home/bjaeger/PufferDrive/experiments/k_scaled_0006/puffer_drive_nht7wfsc/models/model_puffer_drive_007451.pt

source .venv/bin/activate
.venv/bin/puffer eval puffer_drive carla \
    vec.num_envs=16 \
    eval.action_selection=mean \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH}

.venv/bin/puffer eval puffer_drive nuplan_single \
    env.map_dir=/home/shared/data/nuPlan/PufferDrive \
    vec.num_envs=16 \
    eval.action_selection=mean \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH}

#     
end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"