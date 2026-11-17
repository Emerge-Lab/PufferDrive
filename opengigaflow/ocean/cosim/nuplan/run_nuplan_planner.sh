#!/bin/bash
# Run opengigaflow/ocean/cosim/nuplan/planner.py under nuPlan's unmodified closed-loop run_simulation.py for both
# agent-reactivity challenges (nonreactive log replay + reactive IDM).
#
# Usage: CKPT=/path/to/final_model.pt srun --jobid=<job> --overlap bash run_nuplan_planner.sh
#
#   CKPT                   OpenGigaFlow checkpoint .pt (required)
#   SPLIT                  carl_nuplan scenario_filter (default val14_split; filter_pgh = quick Pittsburgh-only check)
#   CHALLENGES             challenge list (default: closed_loop_{nonreactive,reactive}_agents_opengigaflow)
#   THREADS_PER_NODE       caps worker.threads_per_node (ray sizes its pool from the node's CPUs; ~12 GB per worker)
#   WORKER                 nuPlan worker config (default: the devkit's ray_distributed; sequential = one process)
#   CALLBACKS              simulation callbacks (default [simulation_log_callback]; + carl_visualization_callback)
#   LIMIT_TOTAL_SCENARIOS  caps scenario_filter.limit_total_scenarios (unset: the whole SPLIT)
#   SCENARIO_TOKENS        scenario_filter.scenario_tokens
#   GROUP                  output group dir (default $NUPLAN_EXP_ROOT/nuplan_leaderboard_<timestamp>)
#   SLIDING_GOAL_WINDOW    true|false: refill the goal window after every consumed goal (unset = planner yaml, true)
#   STARTUP_ACCEL_JERK_CAP / STARTUP_BRAKE_JERK_CAP / STARTUP_JERK_CAP_SECONDS  ego jerk caps [m/s^3] at the start
#                          (unset = planner yaml, 2 / 2 for 1.5 s; 0 = off)
#   VRU_SIZE_PAD_M         [m] added to pedestrian and bicycle partner boxes in the policy's obs (unset = planner yaml,
#                          0.8; 0 = off)
#   BICYCLE_MIN_SPEED_MPS  [m/s] speed floor of the ego plant (0 = no reverse driving, -2 = Drive's floor)
#   BICYCLE_CENTER_ACCEL_CAP / BICYCLE_CENTER_ACCEL_CAP_MARGIN  CaRL's clip_angular_adjustment and its cap factor
#                          (unset = planner yaml, true / 0.95)
#   STANDSTILL_JERK_DEADBAND  [m/s^3] shadow-env standstill jerk deadband (unset = arch.py's 1.5, 0 = off)
#   OBS_SLOTS_PARTNERS_N   shadow-env partner obs slots (unset = planner yaml, 40)
#   EVAL_PERCEIVED_SIZE_MARGIN_M  [m per side] ego box inflation in the policy's obs (unset = arch.py, 0.2)
#   REWARD_OVERSPEED       overspeed conditioning alpha fed to the policy (unset = arch.py, 0.1)
#   LANE_WIDTH_CLIP_MIN_M / LANE_WIDTH_CLIP_MAX_M  [m] clip the observed lane widths (unset = planner yaml, 2.85 / 4.4;
#                          MAX 0 = off)
#   EXTRA_HYDRA            extra Hydra overrides, space-separated
#   COSIM_OBS_HTML         all|failures (default)|infractions|0: observation replay pages -> $GROUP/obs_html
#   COSIM_OBS_HTML_MAX_SCORE  failures: render scenarios scoring below this (default 0.9)
#   TIDY                   1 (default): scripts/eval/tidy_nuplan_run.py flattens $GROUP; 0 keeps nuPlan's tree
#   PY, PD, NUPLAN_{DATA,MAPS,EXP,DEVKIT}_ROOT, NUPLAN_VAL_DB_DIR, CITY_BIN_DIR  paths (defaults: the cluster layout)
if [ -z "$INSIDE_OPENGIGAFLOW_CONTAINER" ] && [ -f /share/apps/images/slurm-ib-bind-apptainer.sh ]; then
    source /share/apps/images/slurm-ib-bind-apptainer.sh
    export APPTAINERENV_PREPEND_PATH=/share/apps/apptainer/bin:$APPTAINERENV_PREPEND_PATH
    exec singularity exec \
      --bind /scratch/yw4142:/scratch/yw4142 \
      /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
      env INSIDE_OPENGIGAFLOW_CONTAINER=1 bash "$0"
fi
set -u

: "${CKPT:?set CKPT=/path/to/model_xxx.pt}"
PY=${PY:-/scratch/yw4142/venvs/pufferdrive/bin/python}
PD=${PD:-/scratch/yw4142/PufferDrive_cosim}
export NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT:-/scratch/yw4142/datasets/ad/nuplan}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/scratch/yw4142/datasets/ad/nuplan/maps-gpkg}
export NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-/scratch/yw4142/runs/nuplan_exp}
export NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT:-/scratch/yw4142/nuplan-devkit}
mkdir -p "$NUPLAN_EXP_ROOT"

SPLIT=${SPLIT:-val14_split}
CHALLENGES=${CHALLENGES:-"closed_loop_nonreactive_agents_opengigaflow closed_loop_reactive_agents_opengigaflow"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GROUP=${GROUP:-$NUPLAN_EXP_ROOT/nuplan_leaderboard_${TIMESTAMP}}
OBS_HTML=${COSIM_OBS_HTML:-failures}
[ "$OBS_HTML" = "1" ] && OBS_HTML=all

EXTRA_ARGS=()
[ "$OBS_HTML" != "0" ] && EXTRA_ARGS+=("planner.opengigaflow_planner.obs_html_dir=$GROUP/obs_html" "planner.opengigaflow_planner.obs_html_render=false")
CITY_BIN_DIR=${CITY_BIN_DIR:-/scratch/yw4142/datasets/ad/nuplan/maps}
EXTRA_ARGS+=("planner.opengigaflow_planner.city_bin_dir=$CITY_BIN_DIR")
[ -n "${SLIDING_GOAL_WINDOW:-}" ] && EXTRA_ARGS+=("planner.opengigaflow_planner.sliding_goal_window=$SLIDING_GOAL_WINDOW")
[ -n "${STARTUP_JERK_CAP_SECONDS:-}" ] && EXTRA_ARGS+=("planner.opengigaflow_planner.startup_jerk_cap_seconds=$STARTUP_JERK_CAP_SECONDS")
[ -n "${STARTUP_ACCEL_JERK_CAP:-}" ] && EXTRA_ARGS+=("planner.opengigaflow_planner.startup_accel_jerk_cap_mps3=$STARTUP_ACCEL_JERK_CAP")
[ -n "${STARTUP_BRAKE_JERK_CAP:-}" ] && EXTRA_ARGS+=("planner.opengigaflow_planner.startup_brake_jerk_cap_mps3=$STARTUP_BRAKE_JERK_CAP")
[ -n "${VRU_SIZE_PAD_M:-}" ] && EXTRA_ARGS+=("planner.opengigaflow_planner.vru_size_pad_m=$VRU_SIZE_PAD_M")
[ -n "${BICYCLE_MIN_SPEED_MPS:-}" ] && EXTRA_ARGS+=("planner.opengigaflow_planner.bicycle_min_speed_mps=$BICYCLE_MIN_SPEED_MPS")
[ -n "${BICYCLE_CENTER_ACCEL_CAP:-}" ] && EXTRA_ARGS+=("planner.opengigaflow_planner.bicycle_center_accel_cap=$BICYCLE_CENTER_ACCEL_CAP")
[ -n "${BICYCLE_CENTER_ACCEL_CAP_MARGIN:-}" ] && EXTRA_ARGS+=("planner.opengigaflow_planner.bicycle_center_accel_cap_margin=$BICYCLE_CENTER_ACCEL_CAP_MARGIN")
[ -n "${STANDSTILL_JERK_DEADBAND:-}" ] && EXTRA_ARGS+=("++planner.opengigaflow_planner.env_overrides.eval_standstill_jerk_deadband_mps3=$STANDSTILL_JERK_DEADBAND")
[ -n "${OBS_SLOTS_PARTNERS_N:-}" ] && EXTRA_ARGS+=("++planner.opengigaflow_planner.env_overrides.obs_slots_partners_n=$OBS_SLOTS_PARTNERS_N")
[ -n "${EVAL_PERCEIVED_SIZE_MARGIN_M:-}" ] && EXTRA_ARGS+=("++planner.opengigaflow_planner.env_overrides.eval_perceived_size_margin_m=$EVAL_PERCEIVED_SIZE_MARGIN_M")
[ -n "${REWARD_OVERSPEED:-}" ] && EXTRA_ARGS+=("++planner.opengigaflow_planner.env_overrides.reward_overspeed=$REWARD_OVERSPEED")
[ -n "${LANE_WIDTH_CLIP_MIN_M:-}" ] && EXTRA_ARGS+=("++planner.opengigaflow_planner.env_overrides.obs_lane_width_clip_min_m=$LANE_WIDTH_CLIP_MIN_M")
[ -n "${LANE_WIDTH_CLIP_MAX_M:-}" ] && EXTRA_ARGS+=("++planner.opengigaflow_planner.env_overrides.obs_lane_width_clip_max_m=$LANE_WIDTH_CLIP_MAX_M")
[ -n "${WORKER:-}" ] && EXTRA_ARGS+=("worker=$WORKER")
[ -n "${THREADS_PER_NODE:-}" ] && EXTRA_ARGS+=("worker.threads_per_node=$THREADS_PER_NODE")
[ -n "${LIMIT_TOTAL_SCENARIOS:-}" ] && EXTRA_ARGS+=("scenario_filter.limit_total_scenarios=$LIMIT_TOTAL_SCENARIOS")
[ -n "${SCENARIO_TOKENS:-}" ] && EXTRA_ARGS+=("scenario_filter.scenario_tokens=$SCENARIO_TOKENS")
[ -n "${EXTRA_HYDRA:-}" ] && EXTRA_ARGS+=($EXTRA_HYDRA)

cd "$PD"
declare -A STATUS
# no metric_summary_callback: it crashes on read-only pyarrow arrays and would skip csv_main_callback (the results)
for CHALLENGE in $CHALLENGES; do
    echo "[run_nuplan_planner] challenge=$CHALLENGE split=$SPLIT group=$GROUP"
    "$PY" "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
        +simulation=$CHALLENGE \
        scenario_builder=nuplan \
        scenario_builder.data_root="${NUPLAN_VAL_DB_DIR:-$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/val}" \
        scenario_filter=$SPLIT \
        planner=opengigaflow_planner \
        planner.opengigaflow_planner.checkpoint_path="$CKPT" \
        callback="${CALLBACKS:-[simulation_log_callback]}" \
        main_callback="[time_callback, metric_file_callback, metric_aggregator_callback, csv_main_callback]" \
        hydra.searchpath="[pkg://opengigaflow.ocean.cosim.nuplan.config, pkg://carl_nuplan.planning.script.config.common, pkg://carl_nuplan.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
        group="$GROUP" \
        "${EXTRA_ARGS[@]}"
    STATUS[$CHALLENGE]=$?
    echo "[run_nuplan_planner] $CHALLENGE exit ${STATUS[$CHALLENGE]}"
done

if [ "$OBS_HTML" = "all" ]; then
    "$PY" "$PD/scripts/eval/render_obs_html.py" "$GROUP" --max-score 1.01
elif [ "$OBS_HTML" = "failures" ]; then
    "$PY" "$PD/scripts/eval/render_obs_html.py" "$GROUP" --max-score "${COSIM_OBS_HTML_MAX_SCORE:-0.9}" --prune
elif [ "$OBS_HTML" = "infractions" ]; then
    "$PY" "$PD/scripts/eval/render_obs_html.py" "$GROUP" --max-score 0 --prune \
        --metric no_ego_at_fault_collisions --metric drivable_area_compliance \
        --metric driving_direction_compliance --metric ego_is_making_progress
fi
TIDY_KEEP=$OBS_HTML
[ "$OBS_HTML" = "0" ] && TIDY_KEEP=infractions
if [ "${TIDY:-1}" = "1" ]; then
    "$PY" "$PD/scripts/eval/tidy_nuplan_run.py" "$GROUP" \
        --keep "$TIDY_KEEP" --max-score "${COSIM_OBS_HTML_MAX_SCORE:-0.9}" \
        || echo "tidy failed (results above are unaffected)"
fi

echo "=========== SUMMARY ==========="
rc=0
for c in "${!STATUS[@]}"; do
    if [ "${STATUS[$c]}" -eq 0 ]; then s=PASS; else s="FAIL(${STATUS[$c]})"; rc=1; fi
    echo "$s  $c"
done
echo "group -> $GROUP"
[ -f "$GROUP/obs_html/index.html" ] && echo "obs replays -> $GROUP/obs_html/index.html"
[ -f "$GROUP/overview.csv" ] && echo "overview -> $GROUP/overview.csv"
exit $rc
