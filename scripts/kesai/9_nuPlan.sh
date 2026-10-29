#!/bin/bash
#SBATCH --job-name eval_nuplan
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=0
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --output experiments/logs/nuPlan_reactive_%j.out
#SBATCH --error experiments/logs/nuPlan_reactive_%j.err
#SBATCH --partition dev
set -u
source scripts/kesai/cluster_env.sh || exit 1

export PY=$(conda info --base)/envs/carl_nuplan/bin/python
# Overridable env: RUN_DIR (3_train_64GPU_multinode_slurm.sh submits this job with RUN_DIR=<its run dir>), CHALLENGES
RUN_DIR=${RUN_DIR:-$PD/experiments/k_scaled_0045_1000}
case "$RUN_DIR" in /*) ;; *) RUN_DIR=$PD/$RUN_DIR ;; esac # relative to the checkout: $PD is only defined inside the job
# the planner finds config.yaml next to final_model.pt (or one level above a models/*.pt)
export CKPT=$RUN_DIR/final_model.pt
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
# reactive = IDM agents (CLS-R), nonreactive = log-replay agents (CLS-NR)
export CHALLENGES=${CHALLENGES:-closed_loop_reactive_agents_pufferdrive}
case "$CHALLENGES" in
    closed_loop_reactive_agents_pufferdrive) CHALLENGE_TAG="" ;;
    closed_loop_nonreactive_agents_pufferdrive) CHALLENGE_TAG="_nonreactive" ;;
    *) echo "CHALLENGES must be closed_loop_reactive_agents_pufferdrive or closed_loop_nonreactive_agents_pufferdrive"; exit 1 ;;
esac
echo "Evaluating checkpoint: $CKPT"
export CITY_BIN_DIR=$SHARED_DATA/nuplan/PufferDrive
# shell env still exports the pre-rename nuPlan casing; pin the renamed paths here
export NUPLAN_DATA_ROOT=$SHARED_DATA/nuplan
export NUPLAN_MAPS_ROOT=$SHARED_DATA/nuplan/maps
# the runner's default would mkdir a /scratch path of the old cluster (harmless, but a Permission denied line in every kesai log)
export NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-$HOME/nuplan_exp}
# PD first, and drop inherited entries from other PufferDrive checkouts (the login shell still exports
# $HOME/cosim_Puffer): Python would otherwise import that checkout's pufferlib package.
INHERITED_PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v "cosim_Puffer\|/PufferDrive" | paste -sd: -)
export PYTHONPATH=$PD:$CARL_DEVKIT_ROOT:$NUPLAN_DEVKIT_ROOT${INHERITED_PYTHONPATH:+:$INHERITED_PYTHONPATH}
cd "$PD" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export COSIM_OBS_HTML=infractions  # obs replays (what the policy saw) only for collision/offroad/wrong-way/no-progress scenarios; all|failures|infractions|0
# carl_visualization_callback: nuPlan ground-truth video per scenario -> $GROUP/simulation/<challenge>/<ts>/visualization
export CALLBACKS="[simulation_log_callback, carl_visualization_callback]"
export WORKER=ray_distributed
# Per-worker memory is small now (1 policy agent + partner slots, light buffers sized to the scenario), so this is CPU-bound.
# 128 fits the 144-CPU scaleway nodes. A kesai RTX node gives the job 480 CPUs and a worker needs ~1.9 GB: 256 doubles the
# throughput (128 workers used 29 % of the CPUs there, 26 Oct) and leaves half the memory to the page cache of the logs.
export THREADS_PER_NODE=${THREADS_PER_NODE:-$([ "$CLUSTER" = kesai ] && echo 256 || echo 128)}
# route: lane-graph route goals; roadblock: route roadblock centroids; roadblock_lane: roadblock goals on the lane continuing the ego lane
export GOAL_SOURCE=roadblock
# Ablations (see run_nuplan_planner.sh): refill the goal window after every consumed goal; cap the ego's
# accelerating / braking jerk [m/s^3] for the first 1.5 s of each scenario (0 = off).
export SLIDING_GOAL_WINDOW=true
export STARTUP_ACCEL_JERK_CAP=2
export STARTUP_BRAKE_JERK_CAP=2
# pedestrian and bicycle partner boxes grow by this much in length and width in the policy's obs (0 = off): 0.8 puts them strictly above the smallest training vehicle
export VRU_SIZE_PAD_M=0.8
# Shadow-env obs overrides: partner slots (training 20, GIGAFLOW eval 40; 40 real partners max-pooled is OOD in
# sidewalk crowds) and the ego perceived-size margin [m per side] (0.2 = native eval; nuPlan's 2.3 x 5.2 m ego
# reads as 2.7 x 5.6 m). Unset = planner yaml / arch.py defaults (40 / 0.2).
export OBS_SLOTS_PARTNERS_N=40
export EVAL_PERCEIVED_SIZE_MARGIN_M=0.2
# overspeed conditioning alpha (unset = arch.py 0.1 = max of the U(0, 0.1) training range; 0.05 kept the ego at the limit, 0.005 let it speed)
export REWARD_OVERSPEED=0.1
# Lane-width clip [m]: the policy observes lane widths clipped to the training range (CARLA maps 2.85-4.4 m; nuPlan
# city bins have 22% of segments wider than 5 m, up to 50 m). LANE_WIDTH_CLIP_MAX_M=0 turns it off (pre-2026-10-29 evals).
export LANE_WIDTH_CLIP_MIN_M=${LANE_WIDTH_CLIP_MIN_M:-2.85}
export LANE_WIDTH_CLIP_MAX_M=${LANE_WIDTH_CLIP_MAX_M:-4.4}
ABLATION_TAG=""
[ "$SLIDING_GOAL_WINDOW" = "true" ] && ABLATION_TAG="${ABLATION_TAG}_slide"
[ "$STARTUP_ACCEL_JERK_CAP" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_jacc${STARTUP_ACCEL_JERK_CAP}"
[ "$STARTUP_BRAKE_JERK_CAP" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_jbrk${STARTUP_BRAKE_JERK_CAP}"
[ "$VRU_SIZE_PAD_M" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_vrupad${VRU_SIZE_PAD_M}m"
[ -n "${OBS_SLOTS_PARTNERS_N:-}" ] && ABLATION_TAG="${ABLATION_TAG}_slots${OBS_SLOTS_PARTNERS_N}"
[ -n "${EVAL_PERCEIVED_SIZE_MARGIN_M:-}" ] && ABLATION_TAG="${ABLATION_TAG}_margin${EVAL_PERCEIVED_SIZE_MARGIN_M}"
[ -n "${REWARD_OVERSPEED:-}" ] && ABLATION_TAG="${ABLATION_TAG}_overspeed${REWARD_OVERSPEED}"
[ "$LANE_WIDTH_CLIP_MAX_M" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_lanew${LANE_WIDTH_CLIP_MIN_M}-${LANE_WIDTH_CLIP_MAX_M}"
# results live in the model's own eval folder, next to the PufferDrive benchmark evals
export GROUP=$RUN_DIR/eval/nuplan_val14${CHALLENGE_TAG}_${GOAL_SOURCE}${ABLATION_TAG}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}

# NO_TRAIN skips the training-only CUDA extension (_C, never imported by the planner): the cp39 nuPlan env builds only
# the C binding, so nvcc 13.4 on kesai never meets this env's cu128 torch (script 12 builds the same way)
NO_TRAIN=1 PATH="$(dirname "$PY"):$PATH" bash "$PD/scripts/kesai/build_ext_if_changed.sh" "$PD" || exit 1

# The planner must come from this checkout (needs the freshly built C extension above).
IMPORTED_PLANNER=$("$PY" -c "import pufferlib.ocean.cosim.nuplan.planner as p; print(p.__file__)") || { echo "ERROR: planner import failed (see traceback above)"; exit 1; }
case "$IMPORTED_PLANNER" in
    "$PD"/*) echo "planner: $IMPORTED_PLANNER" ;;
    *) echo "ERROR: planner imported from '$IMPORTED_PLANNER', expected $PD"; exit 1 ;;
esac

bash "$PD/pufferlib/ocean/cosim/nuplan/run_nuplan_planner.sh"
