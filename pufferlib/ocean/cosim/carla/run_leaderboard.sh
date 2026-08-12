#!/bin/bash
# Run pufferlib/ocean/cosim/carla/leaderboard_agent.py under CaRL's unmodified
# original_leaderboard evaluator. See README.md "How to run".
#
# Usage (from a GPU allocation's job id — squeue -u $USER to find it):
#   CKPT=/path/to/model_puffer_drive_xxxxxx.pt \
#     srun --jobid=<gpu job> --overlap bash run_leaderboard.sh
#
# Required env:
#   CKPT           PufferDrive checkpoint .pt (experiment dir or models/*.pt file)
# Optional env:
#   ROUTES         routes xml. Default: one longest6 split route (Town01).
#   ROUTES_SUBSET  route id within ROUTES. Default: 0. Set to empty
#                  (ROUTES_SUBSET=) to run every route in ROUTES sequentially
#                  (e.g. point ROUTES at the combined longest6.xml for all 36).
#   OUT            result json path. Default: runs/cosim_leaderboard_<timestamp>/result.json
#   CARLA_PORT     default 2000 (use a distinct port per concurrent instance).
#   COSIM_DEBUG_BEV         dir for one PufferDrive shadow-env BEV mp4 per
#                  route. Default: "$(dirname "$OUT")/bev". Frames are
#                  buffered in memory per route (fine for a debug route,
#                  risky for a long/full-benchmark one).
#   COSIM_DEBUG_CARLA_VIEW  dir for one CARLA chase-cam mp4 per route (an
#                  actual camera sensor on the ego, native 20 Hz tick rate,
#                  streamed to disk frame-by-frame so it's safe for long
#                  routes). Default: "$(dirname "$OUT")/carla_view".
#                  Set either COSIM_DEBUG_BEV= / COSIM_DEBUG_CARLA_VIEW=
#                  (empty) to disable.
#   COSIM_DEVICE / COSIM_DT / COSIM_NUM_AGENTS
#                  forwarded to leaderboard_agent.py as-is if already exported.
if [ -z "$INSIDE_PUFFER_CONTAINER" ]; then
    source /share/apps/images/slurm-ib-bind-apptainer.sh
    export APPTAINERENV_PREPEND_PATH=/share/apps/apptainer/bin:$APPTAINERENV_PREPEND_PATH
    exec singularity exec --nv \
      --bind /opt/slurm:/opt/slurm \
      --bind /scratch/yw4142:/scratch/yw4142 \
      --bind /run/munge:/run/munge \
      --bind /etc/passwd:/etc/passwd \
      --bind /etc/group:/etc/group \
      --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d \
      --bind /etc/vulkan:/etc/vulkan \
      --bind /usr/lib64/libGLX_nvidia.so.0:/usr/lib64/libGLX_nvidia.so.0 \
      --bind /usr/sbin/fuser:/usr/sbin/fuser \
      /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
      env INSIDE_PUFFER_CONTAINER=1 bash "$0"
fi
set -u

: "${CKPT:?set CKPT=/path/to/model_xxx.pt}"
PD=${PD:-/scratch/yw4142/PufferDrive_cosim}
CARL_PY=/scratch/yw4142/conda_envs/carl/bin/python
CARLA_ROOT=/scratch/yw4142/CaRL/CARLA/carla
CARL_WORK_DIR=/scratch/yw4142/CaRL/CARLA
CARLA_PORT=${CARLA_PORT:-2000}
ROUTES=${ROUTES:-$CARL_WORK_DIR/custom_leaderboard/leaderboard/data/longest6.xml}
ROUTES_SUBSET=${ROUTES_SUBSET-0-35}   # ROUTES_SUBSET= (empty) runs every route in ROUTES
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT=${OUT:-/scratch/yw4142/runs/cosim_leaderboard_${TIMESTAMP}/result.json}
mkdir -p "$(dirname "$OUT")"
# PufferDrive shadow-env BEV mp4s and CARLA chase-cam mp4s (one each per
# route), next to OUT by default.
export COSIM_DEBUG_BEV=${COSIM_DEBUG_BEV-"$(dirname "$OUT")/bev"}
export COSIM_DEBUG_CARLA_VIEW=${COSIM_DEBUG_CARLA_VIEW-"$(dirname "$OUT")/carla_view"}

export CARLA_ROOT CARL_WORK_DIR
export PYTHONPATH="$PD:$CARL_WORK_DIR/original_leaderboard/leaderboard:$CARL_WORK_DIR/original_leaderboard/scenario_runner:$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
# route_scenario.py globs $SCENARIO_RUNNER_ROOT/srunner/scenarios/*.py to
# discover scenario classes; leaderboard_agent.py refuses to start without it.
export SCENARIO_RUNNER_ROOT="$CARL_WORK_DIR/original_leaderboard/scenario_runner"

"$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -carla-rpc-port="$CARLA_PORT" &
SERVER_PID=$!
up=1
for i in $(seq 1 60); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "CARLA server process died"; break
    fi
    "$CARL_PY" -c "
import carla, sys
try:
    c = carla.Client('localhost', $CARLA_PORT); c.set_timeout(10.0)
    c.get_world().get_map()
except Exception:
    sys.exit(1)" && up=0 && break
    sleep 5
done
if [ $up -ne 0 ]; then
    echo "FAIL: CARLA server not ready on port $CARLA_PORT"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
echo "[run_leaderboard] CARLA server ready (pid $SERVER_PID, port $CARLA_PORT)"

# Distinct traffic-manager port per instance (default 8000 collides when two
# CARLA servers share a node, killing one evaluator with an rpc bind error).
TM_PORT=${TM_PORT:-$((CARLA_PORT + 6000))}
ARGS=(--routes "$ROUTES"
      --agent "$PD/pufferlib/ocean/cosim/carla/leaderboard_agent.py"
      --agent-config "$CKPT" --checkpoint "$OUT" --track MAP --port "$CARLA_PORT"
      --traffic-manager-port "$TM_PORT")
[ -n "$ROUTES_SUBSET" ] && ARGS+=(--routes-subset "$ROUTES_SUBSET")

echo "[run_leaderboard] routes=$ROUTES subset=${ROUTES_SUBSET:-<all>} ckpt=$CKPT out=$OUT"
"$CARL_PY" -u "$CARL_WORK_DIR/original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py" "${ARGS[@]}"
RC=$?
kill $SERVER_PID 2>/dev/null
echo "[run_leaderboard] evaluator exit $RC; result -> $OUT"
exit $RC
