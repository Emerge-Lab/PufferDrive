#!/bin/bash
# Run pufferlib/ocean/cosim/nuplan/planner.py under nuPlan's unmodified
# closed-loop run_simulation.py: BOTH agent-reactivity challenges (nonreactive
# log-replay + reactive IDM), with both debug videos enabled. See README.md
# "How to run" and planner.py's module docstring.
#
# Usage (from a compute-node allocation's job id -- squeue -u $USER to find it):
#   CKPT=/path/to/model_puffer_drive_xxxxxx.pt \
#     srun --jobid=<job> --overlap bash run_nuplan_planner.sh
#
# Required env:
#   CKPT     PufferDrive checkpoint .pt, or "dummy" for the no-policy wiring test.
# Optional env:
#   SPLIT     carl_nuplan scenario_filter. Default: val14_split (the full
#             closed-loop Val14 benchmark -- ~900 scenarios, "all the
#             scenarios"). filter_pgh is a much smaller Pittsburgh-only
#             filter, useful to validate a new checkpoint quickly first.
#   CHALLENGES  space-separated challenge list. Default: both
#             "closed_loop_nonreactive_agents_pufferdrive
#              closed_loop_reactive_agents_pufferdrive" (log-replay + IDM).
#   THREADS_PER_NODE  caps worker.threads_per_node. ray_distributed sizes its
#             pool from the NODE's cpu count (128 on cs), not the slurm
#             allocation -- 128 nuPlan workers OOMed a 160G job in 34 min
#             (2026-08-09); size this to job_mem / ~12G per worker.
#   WORKER    nuPlan worker config. Default: unset -> falls through to the
#             devkit's own default (ray_distributed), matching
#             scripts/eval/sim_pufferdrive.sh's established convention for a
#             full-benchmark run. NOT verified in this container this
#             session -- only worker=sequential has been exercised end-to-end
#             (one scenario). If ray_distributed misbehaves here, retry with
#             WORKER=sequential (much slower, but proven) while investigating.
#   LIMIT_TOTAL_SCENARIOS  caps scenario_filter.limit_total_scenarios (unset =
#             no cap, SPLIT's own scenario count applies). Handy for a quick
#             sanity check before committing to the full SPLIT.
#   GROUP     output group dir. Default: runs/nuplan_leaderboard_<timestamp>.
#   GOAL_SOURCE=route|gt_map  planner goal source (default route, see pufferdrive_planner.yaml).
#   COSIM_OBS_HTML=all|failures|infractions|0   interactive observation replay (exact policy
#             input/outputs per step, pufferlib.viz HTML) -> $GROUP/obs_html. all: every
#             scenario. failures (default): only scenarios scoring below
#             COSIM_OBS_HTML_MAX_SCORE (default 0.9). infractions: only scenarios with an
#             at-fault collision, drivable-area or driving-direction violation, or no
#             progress. Non-selected replays are deleted after the run. 0: off.
#   (nuPlan's own ground-truth video is always requested via
#    carl_visualization_callback in the callback list below -- one .avi per
#    scenario under $GROUP/simulation/<challenge>/<run_ts>/visualization/)
if [ -z "$INSIDE_PUFFER_CONTAINER" ] && [ -f /share/apps/images/slurm-ib-bind-apptainer.sh ]; then
    source /share/apps/images/slurm-ib-bind-apptainer.sh
    export APPTAINERENV_PREPEND_PATH=/share/apps/apptainer/bin:$APPTAINERENV_PREPEND_PATH
    exec singularity exec \
      --bind /scratch/yw4142:/scratch/yw4142 \
      /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
      env INSIDE_PUFFER_CONTAINER=1 bash "$0"
fi
set -u

: "${CKPT:?set CKPT=/path/to/model_xxx.pt (or CKPT=dummy for the wiring test)}"
PY=${PY:-/scratch/yw4142/venvs/pufferdrive/bin/python}
PD=${PD:-/scratch/yw4142/PufferDrive_cosim}
export NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT:-/scratch/yw4142/datasets/ad/nuplan}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/scratch/yw4142/datasets/ad/nuplan/maps-gpkg}
export NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-/scratch/yw4142/runs/nuplan_exp}
export NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT:-/scratch/yw4142/nuplan-devkit}
mkdir -p "$NUPLAN_EXP_ROOT"

SPLIT=${SPLIT:-val14_split}
CHALLENGES=${CHALLENGES:-"closed_loop_nonreactive_agents_pufferdrive closed_loop_reactive_agents_pufferdrive"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GROUP=${GROUP:-$NUPLAN_EXP_ROOT/nuplan_leaderboard_${TIMESTAMP}}
OBS_HTML=${COSIM_OBS_HTML:-failures}
[ "$OBS_HTML" = "1" ] && OBS_HTML=all

EXTRA_ARGS=()
[ "$OBS_HTML" != "0" ] && EXTRA_ARGS+=("planner.pufferdrive_planner.obs_html_dir=$GROUP/obs_html" "planner.pufferdrive_planner.obs_html_render=false")
CITY_BIN_DIR=${CITY_BIN_DIR:-/scratch/yw4142/datasets/ad/nuplan/maps}
EXTRA_ARGS+=("planner.pufferdrive_planner.city_bin_dir=$CITY_BIN_DIR")
[ -n "${GOAL_SOURCE:-}" ] && EXTRA_ARGS+=("planner.pufferdrive_planner.goal_source=$GOAL_SOURCE")
[ -n "${WORKER:-}" ] && EXTRA_ARGS+=("worker=$WORKER")
[ -n "${THREADS_PER_NODE:-}" ] && EXTRA_ARGS+=("worker.threads_per_node=$THREADS_PER_NODE")
[ -n "${LIMIT_TOTAL_SCENARIOS:-}" ] && EXTRA_ARGS+=("scenario_filter.limit_total_scenarios=$LIMIT_TOTAL_SCENARIOS")
[ -n "${SCENARIO_TOKENS:-}" ] && EXTRA_ARGS+=("scenario_filter.scenario_tokens=$SCENARIO_TOKENS")
[ -n "${EXTRA_HYDRA:-}" ] && EXTRA_ARGS+=($EXTRA_HYDRA)

cd "$PD"
declare -A STATUS
# main_callback below omits metric_summary_callback (PDF histograms): it
# crashes on a read-only numpy array from pyarrow-backed parquet columns in
# this nuplan-devkit checkout (nuboard_histogram_utils.py), which aborts the
# whole main_callback chain and skips csv_main_callback -- the actual
# results. csv_main_callback reads aggregator parquet files directly and
# doesn't need metric_summary_callback's output.
for CHALLENGE in $CHALLENGES; do
    echo "[run_nuplan_planner] challenge=$CHALLENGE split=$SPLIT group=$GROUP"
    "$PY" "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
        +simulation=$CHALLENGE \
        scenario_builder=nuplan \
        scenario_builder.data_root="${NUPLAN_VAL_DB_DIR:-$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/val}" \
        scenario_filter=$SPLIT \
        planner=pufferdrive_planner \
        planner.pufferdrive_planner.checkpoint_path="$CKPT" \
        callback="${CALLBACKS:-[simulation_log_callback, carl_visualization_callback]}" \
        main_callback="[time_callback, metric_file_callback, metric_aggregator_callback, csv_main_callback]" \
        hydra.searchpath="[pkg://pufferlib.ocean.cosim.nuplan.config, pkg://carl_nuplan.planning.script.config.common, pkg://carl_nuplan.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
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

echo "=========== SUMMARY ==========="
rc=0
for c in "${!STATUS[@]}"; do
    if [ "${STATUS[$c]}" -eq 0 ]; then s=PASS; else s="FAIL(${STATUS[$c]})"; rc=1; fi
    echo "$s  $c"
done
echo "group -> $GROUP"
[ -f "$GROUP/obs_html/index.html" ] && echo "obs replays -> $GROUP/obs_html/index.html"
exit $rc
