#!/bin/bash
#SBATCH --job-name=train_puffer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task=18
#SBATCH --output=/home/bjaeger/PufferDrive/experiments/k_nightly_0008/puffer_drive_kxgplcb0/eval/eval_nuPlan_%a_%A.out
#SBATCH --error=/home/bjaeger/PufferDrive/experiments/k_nightly_0008/puffer_drive_kxgplcb0/eval/eval_nuPlan_%a_%A.err
#SBATCH --partition=dev

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

RESULTS_PATH=/home/bjaeger/PufferDrive/experiments/k_nightly_0008/puffer_drive_kxgplcb0/eval
MODEL_PATH=/home/bjaeger/PufferDrive/experiments/k_nightly_0008/puffer_drive_kxgplcb0/puffer_drive_kxgplcb0.pt

source .venv/bin/activate
.venv/bin/puffer eval puffer_drive \
    --eval-simulation replay \
    --render 1 \
    --render-backend obs_html \
    eval.validation_defaults.action_selection=mean \
    eval.validation_replay.env.map_dir=/home/shared/data/nuPlan/PufferDrive \
    eval.validation_defaults.env.traffic_light_behavior=1 \
    env.min_goal_spacing=30.0 \
    env.max_goal_spacing=30.0 \
    load_model_path=${MODEL_PATH} \
    +render_results_dir=${RESULTS_PATH}

.venv/bin/puffer eval puffer_drive \
    --eval-simulation gigaflow \
    --render 1 \
    --render-backend obs_html \
    eval.validation_defaults.eval.num_scenarios=1000 \
    load_model_path=${MODEL_PATH} \
    eval.validation_defaults.action_selection=mean \
    eval.validation_gigaflow.env.min_agents_per_env=50 \
    eval.validation_gigaflow.env.max_agents_per_env=50 \
    eval.validation_gigaflow.env.scenario_length=6000 \
    eval.validation_defaults.env.traffic_light_behavior=1 \
    env.min_goal_spacing=30.0 \
    env.max_goal_spacing=30.0 \
    +render_results_dir=${RESULTS_PATH}



# 50 agents per scenario, 1k scenario, 6k steps

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
