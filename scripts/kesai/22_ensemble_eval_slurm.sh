#!/bin/bash
#SBATCH --job-name ensemble_eval
#SBATCH --nodes 1                        # the self-play eval uses every node of the allocation; more nodes for bigger ensembles
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:8                     # GPUs per node
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --mem=0
#SBATCH --time 1-00:00
#SBATCH --output experiments/logs/ensemble_eval_%j.out
#SBATCH --error experiments/logs/ensemble_eval_%j.err
#SBATCH --partition dev

source scripts/kesai/cluster_env.sh || exit 1

# Evaluates an ensemble of trained runs with the evals of 3_train / 18_train. scripts/eval/build_ensemble_checkpoint.py
# packs the members' final_model.pt into $PD/experiments/$ENSEMBLE_NAME/final_model.pt next to the first member's
# config.yaml, so every eval takes the folder like a training run: at every policy step the ensemble executes the mean of
# the members' mean actions (one forward pass per member; mode / sampled actions and the logged probabilities come from
# the members' averaged distribution).
# The ensemble has no wandb run: the chained evals get WANDB_REPORT=0 (12_alpasim.sh has no such switch; its report step
# fails harmlessly) and the self-play eval runs with wandb=False; results live in experiments/$ENSEMBLE_NAME/eval/.
echo "START TIME: $(date)"
start=$(date +%s)

ENSEMBLE_NAME=k_scaled_0073_0074_ensemble
MEMBER_RUNS=(k_scaled_0073_1000 k_scaled_0073_2000 k_scaled_0074_1000 k_scaled_0074_2000)   # run folders under $PD/experiments, each with a final_model.pt
echo "ensemble ${ENSEMBLE_NAME} of ${MEMBER_RUNS[*]}"

DATA_DIR=$PD/experiments/${ENSEMBLE_NAME}
MODEL_PATH=${DATA_DIR}/final_model.pt
MEMBER_CHECKPOINTS=()
for member_run in "${MEMBER_RUNS[@]}"; do
    member_checkpoint=$PD/experiments/${member_run}/final_model.pt
    if [ ! -f "${member_checkpoint}" ]; then
        echo "missing member checkpoint ${member_checkpoint}"
        exit 1
    fi
    MEMBER_CHECKPOINTS+=("${member_checkpoint}")
done

# Thread limit limits CPU thrashing across worker environments
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Hand-written CUDA slot-encoder kernels instead of the Triton ones (same math); triton to compare. Passed per command,
# never exported: sbatch copies this shell's environment into the eval jobs below, whose interpreters lack cuda-bindings.
SLOT_ENCODER_BACKEND=cuda
echo "slot encoder backend: ${SLOT_ENCODER_BACKEND}"

source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh $PD || exit 1

echo "Building ${MODEL_PATH}"
.venv/bin/python scripts/eval/build_ensemble_checkpoint.py \
    --output-dir ${DATA_DIR} \
    --run-name ${ENSEMBLE_NAME} \
    "${MEMBER_CHECKPOINTS[@]}" || exit 1

# nuPlan (reactive and non-reactive), longest6 (v2 and v1) and AlpaSim evals need one 8-GPU node each: submit them as their own jobs before the long self-play eval below so they queue while it runs.
echo "Submitting nuPlan reactive eval for ${MODEL_PATH}"
WANDB_REPORT=0 RUN_DIR=${DATA_DIR} sbatch scripts/kesai/9_nuPlan.sh \
    || echo "nuPlan eval submission failed; run by hand: WANDB_REPORT=0 RUN_DIR=${DATA_DIR} sbatch scripts/kesai/9_nuPlan.sh"
echo "Submitting nuPlan non-reactive eval for ${MODEL_PATH}"
WANDB_REPORT=0 RUN_DIR=${DATA_DIR} sbatch scripts/kesai/13_nuPlan_nonreactive.sh \
    || echo "nuPlan non-reactive eval submission failed; run by hand: WANDB_REPORT=0 RUN_DIR=${DATA_DIR} sbatch scripts/kesai/13_nuPlan_nonreactive.sh"
echo "Submitting longest6 v2 eval for ${MODEL_PATH}"
WANDB_REPORT=0 RUN_DIR=${DATA_DIR} sbatch scripts/kesai/11_carla_longest6_v2.sh \
    || echo "longest6 v2 eval submission failed; run by hand: WANDB_REPORT=0 RUN_DIR=${DATA_DIR} sbatch scripts/kesai/11_carla_longest6_v2.sh"
echo "Submitting longest6 v1 eval for ${MODEL_PATH}"
WANDB_REPORT=0 RUN_DIR=${DATA_DIR} sbatch scripts/kesai/14_carla_longest6_v1.sh \
    || echo "longest6 v1 eval submission failed; run by hand: WANDB_REPORT=0 RUN_DIR=${DATA_DIR} sbatch scripts/kesai/14_carla_longest6_v1.sh"
echo "Submitting AlpaSim eval for ${MODEL_PATH}"
RUN_DIR=${DATA_DIR} sbatch scripts/kesai/12_alpasim.sh \
    || echo "AlpaSim eval submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/12_alpasim.sh"

# parallel_eval runs 8 srun shards per node (one GPU and an eighth of the CPUs each, 60 blocking workers like
# training) so the eval uses every GPU and thread of the allocation; 60 workers x 5 scenarios = 300 scenarios per 9000-step episode.
echo "Follow-up evals submitted, evaluating ${MODEL_PATH}"
PUFFER_SLOT_ENCODER_BACKEND=${SLOT_ENCODER_BACKEND} .venv/bin/python scripts/parallel_eval.py carla \
    --total-scenarios 80000 \
    --num-nodes ${SLURM_JOB_NUM_NODES} \
    --shards-per-node 8 \
    env.map_dir=$PD/pufferlib/resources/drive/binaries/carla \
    vec.num_envs=60 \
    eval.reward_comfort=0.0 \
    eval.reward_lane_center=0.0075 \
    env.eval_perceived_size_margin_m=0.2 \
    eval.min_goal_spacing=20 \
    eval.max_goal_spacing=200 \
    env.disable_red_light_infractions=1 \
    env.disable_stop_sign_infractions=1 \
    env.traffic_light_junction_phases=0 \
    env.eval_standstill_jerk_deadband_mps3=1.5 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    eval.output_name=${ENSEMBLE_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=False

end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
