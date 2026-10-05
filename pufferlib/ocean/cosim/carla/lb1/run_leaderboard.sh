#!/bin/bash
# Run lb1/leaderboard_agent.py under carla_garage's leaderboard_evaluator_local.py (CARLA Leaderboard 1.0,
# CARLA 0.9.10.1): start a CARLA server, evaluate one routes xml, stop the server. One machine, one GPU;
# scripts/kesai/14_carla_longest6_v1.sh is the cluster version over all 36 longest6 routes.
#
# Usage:
#   CKPT=experiments/<run>/final_model.pt bash pufferlib/ocean/cosim/carla/lb1/run_leaderboard.sh
#
# Required env:
#   CKPT             PufferDrive checkpoint .pt (config.yaml beside it or one level up) or an experiment dir
# Optional env:
#   ROUTES           routes xml. Default: longest6_split/longest_weathers_0.xml (one Town01 route).
#   SCENARIOS        scenario json. Default: eval_scenarios.json (the benchmark); no_scenarios.json = empty.
#   REPETITIONS      default 1
#   OUT              result json. Default: runs/cosim_leaderboard_v1_<timestamp>/result.json
#   CARLA_PORT       default 2000 (distinct per concurrent server); TM_PORT default CARLA_PORT+6000
#   CARLA_WINDOW     1: rendering window on $DISPLAY; default 0: offscreen OpenGL (DISPLAY unset, -opengl)
#   CARLA_ROOT       CARLA 0.9.10.1 install.            Default: ~/ordnung/internal/CARLA_0.9.10
#   GARAGE_WORK_DIR  carla_garage checkout.             Default: ~/ordnung/internal/carla_garage_1
#   GARAGE_PY        python of the `garage` conda env.  Default: ~/miniconda3/envs/garage/bin/python
#   COSIM_SERVER_PYTHON  python of the PufferDrive venv for policy_server.py. Default: $PD/.venv/bin/python
#   PD               PufferDrive checkout. Default: the one holding this script.
#   LIBTIFF5_COMPAT  1 (default): the 0.9.10 egg links libtiff.so.5. When `import carla` fails, the garage env's
#                    own lib dir is tried first (`conda install -n garage "libtiff<4.5"` puts one there), then
#                    the system libtiff.so.6 is linked under that name inside $OUT's folder (0 = never link).
#   COSIM_*          forwarded to the agent / policy server as-is (see lb1/leaderboard_agent.py, shadow_ego.py);
#                    COSIM_DEBUG_CARLA_VIEW defaults to "$(dirname "$OUT")/carla_view" (empty disables).
set -u

: "${CKPT:?set CKPT=/path/to/final_model.pt}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PD=${PD:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}
CARLA_ROOT=${CARLA_ROOT:-$HOME/ordnung/internal/CARLA_0.9.10}
GARAGE_WORK_DIR=${GARAGE_WORK_DIR:-$HOME/ordnung/internal/carla_garage_1}
GARAGE_PY=${GARAGE_PY:-$HOME/miniconda3/envs/garage/bin/python}
export COSIM_SERVER_PYTHON=${COSIM_SERVER_PYTHON:-$PD/.venv/bin/python}
CARLA_PORT=${CARLA_PORT:-2000}
TM_PORT=${TM_PORT:-$((CARLA_PORT + 6000))}
CARLA_WINDOW=${CARLA_WINDOW:-0}
ROUTES=${ROUTES:-$GARAGE_WORK_DIR/leaderboard/data/longest6_split/longest_weathers_0.xml}
SCENARIOS=${SCENARIOS:-$GARAGE_WORK_DIR/leaderboard/data/scenarios/eval_scenarios.json}
REPETITIONS=${REPETITIONS:-1}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT=${OUT:-$PD/runs/cosim_leaderboard_v1_${TIMESTAMP}/result.json}
OUT_DIR=$(dirname "$OUT")
mkdir -p "$OUT_DIR"
export COSIM_DEBUG_CARLA_VIEW=${COSIM_DEBUG_CARLA_VIEW-"$OUT_DIR/carla_view"}
EVALUATOR=$GARAGE_WORK_DIR/leaderboard/leaderboard/leaderboard_evaluator_local.py
CARLA_EGG=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
for path in "$CKPT" "$GARAGE_PY" "$COSIM_SERVER_PYTHON" "$CARLA_ROOT/CarlaUE4.sh" "$CARLA_EGG" "$EVALUATOR" "$ROUTES" "$SCENARIOS"; do
    [ -e "$path" ] || { echo "missing $path"; exit 1; }
done

carla_imports() { PYTHONPATH="$CARLA_EGG" "$GARAGE_PY" -c "import carla" >/dev/null 2>&1; }
# the 0.9.10 egg links libtiff.so.5, which newer distros no longer ship (Ubuntu 24.04: libtiff.so.6 only)
if ! carla_imports; then
    GARAGE_LIB=$(cd "$(dirname "$GARAGE_PY")/../lib" 2>/dev/null && pwd)
    if [ -n "$GARAGE_LIB" ] && [ -e "$GARAGE_LIB/libtiff.so.5" ]; then
        export LD_LIBRARY_PATH="$GARAGE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[run_leaderboard_v1] libtiff.so.5 from the garage env: $GARAGE_LIB"
    fi
fi
if ! carla_imports && [ "${LIBTIFF5_COMPAT:-1}" = "1" ]; then
    SYSTEM_LIBTIFF=$( (ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null) | awk '/libtiff.so.6 /{print $NF; exit}')
    if [ -n "$SYSTEM_LIBTIFF" ]; then
        mkdir -p "$OUT_DIR/compat_lib" && ln -sf "$SYSTEM_LIBTIFF" "$OUT_DIR/compat_lib/libtiff.so.5"
        export LD_LIBRARY_PATH="$OUT_DIR/compat_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[run_leaderboard_v1] libtiff.so.5 missing: using $SYSTEM_LIBTIFF via $OUT_DIR/compat_lib"
    fi
fi
carla_imports || { echo "ERROR: $GARAGE_PY cannot import the CARLA egg:"; PYTHONPATH="$CARLA_EGG" "$GARAGE_PY" -c "import carla"; exit 1; }

export CARLA_ROOT GARAGE_WORK_DIR
export SCENARIO_RUNNER_ROOT=$GARAGE_WORK_DIR/scenario_runner
export LEADERBOARD_ROOT=$GARAGE_WORK_DIR/leaderboard
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG:$SCENARIO_RUNNER_ROOT:$LEADERBOARD_ROOT:${PYTHONPATH:-}"
# the garage evaluator names the route records after $ROUTES; BENCHMARK=longest6 fills every spawn point
export ROUTES BENCHMARK=${BENCHMARK:-longest6} DATAGEN=0

if [ "$CARLA_WINDOW" = "1" ]; then
    "$CARLA_ROOT/CarlaUE4.sh" -opengl -nosound -carla-rpc-port="$CARLA_PORT" -carla-streaming-port=$((CARLA_PORT + 1)) \
        > "$OUT_DIR/carla_server.log" 2>&1 &
else
    DISPLAY= "$CARLA_ROOT/CarlaUE4.sh" -opengl -nosound -carla-rpc-port="$CARLA_PORT" -carla-streaming-port=$((CARLA_PORT + 1)) \
        > "$OUT_DIR/carla_server.log" 2>&1 &
fi
SERVER_PID=$!
up=1
for _ in $(seq 1 60); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "CARLA server process died"; break
    fi
    "$GARAGE_PY" -c "
import carla, sys
try:
    c = carla.Client('localhost', $CARLA_PORT); c.set_timeout(10.0)
    c.get_world().get_map()
except Exception:
    sys.exit(1)" 2>/dev/null && up=0 && break
    sleep 5
done
if [ $up -ne 0 ]; then
    echo "FAIL: CARLA server not ready on port $CARLA_PORT (see $OUT_DIR/carla_server.log)"
    pkill -9 -P $SERVER_PID 2>/dev/null; kill -9 $SERVER_PID 2>/dev/null
    exit 1
fi
echo "[run_leaderboard_v1] CARLA 0.9.10 server ready (pid $SERVER_PID, port $CARLA_PORT)"

echo "[run_leaderboard_v1] routes=$ROUTES scenarios=$SCENARIOS ckpt=$CKPT out=$OUT"
"$GARAGE_PY" -u "$EVALUATOR" --routes "$ROUTES" --scenarios "$SCENARIOS" --repetitions "$REPETITIONS" \
    --agent "$SCRIPT_DIR/leaderboard_agent.py" --agent-config "$CKPT" --checkpoint "$OUT" --track MAP \
    --port "$CARLA_PORT" --trafficManagerPort "$TM_PORT" --timeout 600
RC=$?
# children first: killing the wrapper shell first reparents the UE4 binary and leaves it running
pkill -9 -P $SERVER_PID 2>/dev/null; kill -9 $SERVER_PID 2>/dev/null
echo "[run_leaderboard_v1] evaluator exit $RC; result -> $OUT"
exit $RC
