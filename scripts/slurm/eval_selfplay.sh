#!/bin/bash
#SBATCH --job-name eval_selfplay
#SBATCH --nodes 1                        # every node runs GPUS_PER_NODE shards
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:8
#SBATCH --exclusive                      # whole nodes: the env workers use every CPU
#SBATCH --mem=0
#SBATCH --time 12:00:00
#SBATCH --output experiments/logs/eval_selfplay_%j.out
#SBATCH --error experiments/logs/eval_selfplay_%j.err
##SBATCH --partition <partition>         # site-specific options: uncomment and set, or pass them to sbatch
# Self-play benchmark of a run's final_model.pt, one shard per GPU of the allocation (scripts/parallel_eval.py); results go to
# $RUN_DIR/eval (or $EVAL_OUTPUT_DIR_NAME). Submit from the checkout root, once experiments/logs exists:
#   RUN_DIR=experiments/<run> sbatch scripts/slurm/eval_selfplay.sh
# Settings, from the environment or edited here:
#   RUN_DIR                      the training run (holds final_model.pt and config.yaml)
#   GPUS_PER_NODE (8)            shards per node, one GPU each
#   WORKERS_PER_GPU              env workers per shard (default: the node's CPUs / GPUS_PER_NODE)
#   MAP_DIR                      maps instead of the benchmark's; the benchmark reads only the folder's first num_maps sorted maps
#   TOTAL_SCENARIOS, SCENARIO_LENGTH  instead of the benchmark's
#   EVAL_OUTPUT_DIR_NAME         results folder inside the run instead of eval
#   RENDER_FILTER (all_infractions), CAPTURE_OBSERVATIONS (true)  replay pages of the failed episodes; RENDER_FILTER= for none
#   WANDB_REPORT (0)             1: km per infraction -> the run's final_eval/selfplay_km_per_infraction in wandb
RUN_DIR=${RUN_DIR:-experiments/my_run}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-}
MAP_DIR=${MAP_DIR:-}
TOTAL_SCENARIOS=${TOTAL_SCENARIOS:-}
SCENARIO_LENGTH=${SCENARIO_LENGTH:-}
EVAL_OUTPUT_DIR_NAME=${EVAL_OUTPUT_DIR_NAME:-}
RENDER_FILTER=${RENDER_FILTER-all_infractions}
CAPTURE_OBSERVATIONS=${CAPTURE_OBSERVATIONS:-true}
WANDB_REPORT=${WANDB_REPORT:-0}

source scripts/slurm/job_env.sh || exit 1

case "$RUN_DIR" in /*) ;; *) RUN_DIR=$PD/$RUN_DIR ;; esac
MODEL_PATH=$RUN_DIR/final_model.pt
[ -f "$MODEL_PATH" ] || { echo "missing checkpoint $MODEL_PATH (set RUN_DIR)" >&2; exit 1; }
[ -z "$MAP_DIR" ] || [ -d "$MAP_DIR" ] || { echo "missing map dir $MAP_DIR" >&2; exit 1; }
WORKERS_PER_GPU=${WORKERS_PER_GPU:-$((SLURM_CPUS_PER_TASK / GPUS_PER_NODE))}

echo "START TIME: $(date)"
start=$(date +%s)
echo "run $RUN_DIR  maps ${MAP_DIR:-benchmark default}  output dir ${EVAL_OUTPUT_DIR_NAME:-eval}  scenarios ${TOTAL_SCENARIOS:-benchmark default}  scenario_length ${SCENARIO_LENGTH:-benchmark default}"

# one BLAS/OpenMP thread per process: the env workers would otherwise thrash the CPUs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# one srun step per GPU, each with its share of the node's CPUs
.venv/bin/python scripts/parallel_eval.py selfplay \
    ${TOTAL_SCENARIOS:+--total-scenarios "$TOTAL_SCENARIOS"} \
    --num-nodes "$SLURM_JOB_NUM_NODES" \
    --shards-per-node "$GPUS_PER_NODE" \
    ${MAP_DIR:+env.map_dir="$MAP_DIR"} \
    vec.num_envs="$WORKERS_PER_GPU" \
    ${RENDER_FILTER:+eval.render_filter="$RENDER_FILTER"} \
    eval.capture_observations="$CAPTURE_OBSERVATIONS" \
    ${SCENARIO_LENGTH:+env.scenario_length="$SCENARIO_LENGTH"} \
    ${EVAL_OUTPUT_DIR_NAME:+eval.output_dir_name="$EVAL_OUTPUT_DIR_NAME"} \
    eval.output_name="$(basename "$RUN_DIR")" \
    load_model_path="$MODEL_PATH" \
    wandb="$WANDB_REPORT"
EVAL_STATUS=$?

end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start))"
exit $EVAL_STATUS
