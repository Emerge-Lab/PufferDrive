#!/bin/bash
#SBATCH --job-name train_puffer
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 2-00:00
#SBATCH --gres gpu:8
#SBATCH --mem 1024G
#SBATCH --cpus-per-task 224
#SBATCH --output /lustre/scwpod02/client/kyutai/kesai/bernhard/PufferDrive/experiments/k_005_longrun_300B/log_%a_%A.out
#SBATCH --error /lustre/scwpod02/client/kyutai/kesai/bernhard/PufferDrive/experiments/k_005_longrun_300B/log_%a_%A.err
#SBATCH --partition kyutai

# print info about current job
echo "START TIME: $(date)"
start=$(date +%s)

export RUN_NAME=k_006_longrun_300B
echo ${RUN_NAME}

# TODO could try to tune these. 1 Is probably best since Puffer parallelizes across all cores.
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source .venv/bin/activate
torchrun --standalone --nnodes=1 --nproc-per-node=8 --max_restarts=0 --start-method spawn -m pufferlib.pufferl train puffer_drive --wandb --train.data-dir /lustre/scwpod02/client/kyutai/kesai/bernhard/PufferDrive/experiments/${RUN_NAME} \
    --wandb --wandb-project pufferdrive --wandb-group cluster --train.checkpoint-interval 1000 --run-name ${RUN_NAME} \
    --train.total-timesteps 300000000000 \
    --train.max-minibatch-size 100000 \
    --train.minibatch-size 100000 \
    --vec.num-envs 48 \
    --vec.num-workers 24 \
    --eval.validation-replay.env.map-dir /lustre/scwpod02/client/kyutai/kesai/data/nuPlan/PufferDrive \
    --eval.behaviors-full-dir.env.map-dir /lustre/scwpod02/client/kyutai/kesai/data/nuPlan/PufferDrive \
    --env.dt 0.3 \
    --env.max-agents-per-env 120 \
    --env.num-agents 4096 \
    --env.obs-boundary-stride 1 \
    --env.obs-dropout-boundary 0.3 \
    --env.obs-dropout-lane 0.2 \
    --env.obs-lane-stride 1 \
    --env.obs-norm-goal-offset-m 200.0 \
    --env.obs-norm-veh-length-m 10.0 \
    --env.obs-norm-veh-width-m 5.0 \
    --env.obs-norm-xy-offset-m 150.0 \
    --env.obs-range-partner-m 150.0 \
    --env.obs-range-road-behind-m 40.0 \
    --env.obs-range-road-front-m 150.0 \
    --env.obs-range-road-side-m 50.0 \
    --env.obs-range-traffic-control-m 150.0 \
    --env.obs-slots-boundary-n 40 \
    --env.obs-slots-lane-n 60 \
    --env.obs-slots-partners-n 12 \
    --env.resample-frequency 256000 \
    --env.reward-goal 0.5 \
    --env.reward-lane-center 0.005 \
    --env.scenario-length 2560 \
    --env.use-map-cache 1 \
    --policy.backbone-hidden-size 1024 \
    --policy.backbone-num-layers 3 \
    --policy.ego-input-size 128 \
    --policy.lane-input-size 128 \
    --policy.partner-input-size 128 \
    --policy.shared-network false \
    --policy.traffic-control-input-size 128 \
    --train.checkpoint-interval 50 \
    --train.compile true \
    --train.normalize-rewards false \
    --train.precision bfloat16 \
    --train.seed 4 \
    --train.update-epochs 3 \
    --policy.boundary-input-size 128 \
    --tb

end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"