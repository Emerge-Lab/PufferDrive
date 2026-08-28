#!/bin/bash
#SBATCH --job-name eval_puffer_8node
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:8
#SBATCH --cpus-per-task 144
#SBATCH --mem=1007G
#SBATCH --time 1-00:00
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/eval_%a_%A.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/eval_%a_%A.err
#SBATCH --partition dev

echo "START TIME: $(date)"
start=$(date +%s)

export RUN_NAME=k_scaled_0030_1000
echo ${RUN_NAME}

export MODEL_PATH=/home/bjaeger/PufferDrive/experiments/${RUN_NAME}/final_model.pt
echo ${MODEL_PATH}

# Thread limit limits CPU thrashing across worker environments
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh /home/bjaeger/PufferDrive || exit 1

if [ ! -f ${MODEL_PATH} ]; then
    echo "${MODEL_PATH} is missing; nothing to evaluate."
    exit 1
fi

# parallel_eval places one shard per allocated node via srun, so each shard's 64
# env workers get a full node's cores instead of sharing the batch host.
.venv/bin/python scripts/parallel_eval.py carla \
    --total-scenarios 1000 \
    --num-nodes 8 \
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla_hole_fixes_tl \
    vec.num_envs=64 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    eval.reward_comfort=0.0 \
    eval.reward_lane_center=0.0075 \
    eval.dt=0.0667 \
    eval.base_max_speed_mps=20.0 \
    eval.obs_slots_partners_n=16 \
    eval.goal_radius=10.0 \
    eval.goal_source=route \
    env.goal_heading_max_deg=60.0 \
    env.eval_perceived_size_margin_m=0.2 \
    eval.min_goal_spacing=20 \
    eval.max_goal_spacing=30 \
    eval.goal_speed=3.0 \
    env.max_speed_mps=13.33 \
    env.disable_red_light_infractions=0 \
    eval.output_name=${RUN_NAME}_short2 \
    load_model_path=${MODEL_PATH} \
    wandb=True
#     eval.obs_slots_partners_n=40 \
#     eval.goal_regen_mode=rolling \

# No srun: single-node eval, run here on the batch host.
.venv/bin/puffer eval puffer_drive nuplan_multi \
    env.map_dir=/home/shared/data/nuPlan/PufferDrive \
    vec.num_envs=64 \
    eval.max_sdc_replay_workers=64 \
    eval.num_agents=300 \
    eval.reward_comfort=0.0 \
    eval.reward_lane_center=0.0075 \
    eval.dt=0.0667 \
    eval.base_max_speed_mps=20.0 \
    eval.obs_slots_partners_n=16 \
    eval.goal_radius=10.0 \
    eval.goal_source=route \
    env.goal_heading_max_deg=60.0 \
    env.eval_perceived_size_margin_m=0.2 \
    eval.min_goal_spacing=20 \
    eval.max_goal_spacing=30 \
    eval.goal_speed=3.0 \
    env.max_speed_mps=13.33 \
    eval.disable_red_light_infractions=0 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    eval.output_name=${RUN_NAME}_medium \
    load_model_path=${MODEL_PATH} \
    wandb=True

end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
