#!/bin/bash
#SBATCH --job-name eval_longest6
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=1007G
#SBATCH --cpus-per-task 144
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/log_%j.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/log_%j.err
#SBATCH --partition dev
# CARLA longest6 evaluation of a PufferDrive checkpoint through CaRL's unmodified original_leaderboard
# (pufferlib/ocean/cosim/carla/leaderboard_agent.py), the CARLA counterpart of 9_nuPlan.sh.
#
# One CARLA server + one evaluator per GPU (1-8: whatever the allocation holds, e.g. `sbatch --gres gpu:2`);
# the 36 routes are dealt round-robin over the GPUs, every route gets its own server (restarted per route,
# CaRL's own recipe), crashed routes are retried, and the per-route result jsons are aggregated at the end
# with CaRL's tools/result_parser.py (results.csv) plus the nuPlan-style HTML report ($OUT/report/index.html)
# and the obs replay gallery of all routes ($OUT/obs_html/index.html, as in 9_nuPlan.sh).
#
# Overridable env: RUN_DIR, ROUTES, SCENARIOS (0 = pufferlib's scenario-free longest6), REPETITIONS,
# ROUTE_SUBSET ("0 5 17": only these route ids), NUM_GPUS, PORT_BASE (CARLA rpc port of gpu 0; gpu w uses
# PORT_BASE+50w, TM PORT_BASE+6000+50w), LOGGING (0, default: scores only, CARLA runs without rendering
# (-nullrhi, which segfaults as soon as any sensor is requested, so no camera and no telemetry, whose CaRL
# criteria attach a collision sensor); 1: chase-cam video, telemetry, world log and the HTML report per
# route), OBS_HTML (1 = also the interactive obs replay per route, rendered into the one gallery folder
# $OUT/obs_html at the end, large; needs LOGGING=1), REPORT (0 = skip the HTML report),
# MAX_ATTEMPTS, CARLA_ROOT, CARL_WORK_DIR, PY, PD, COSIM_MAX_SPEED_MPS (ego speed cap, default 30),
# COSIM_ZERO_PARTNER_STOPPED_TIME (default 1: partners' stopped-time obs held at 0; 0 = real stopped times),
# COSIM_PEDESTRIAN_MIN_SIZE_M (default 0: true CARLA walker boxes; 0.8 = the training spawn floor).
set -u

export PD=${PD:-/home/bjaeger/PufferDrive}
export PY=${PY:-/home/bjaeger/miniconda3/envs/carl/bin/python}   # cp310: CARLA 0.9.15 ships no newer wheel
export CARLA_ROOT=${CARLA_ROOT:-/home/bjaeger/CARLA_0.9.15}
export CARL_WORK_DIR=${CARL_WORK_DIR:-/home/bjaeger/CaRL/CARLA}
RUN_DIR=${RUN_DIR:-/home/bjaeger/PufferDrive/experiments/k_scaled_0040_1000}
# the agent finds config.yaml next to final_model.pt (or one level above a models/*.pt)
export CKPT=$RUN_DIR/final_model.pt
SCENARIOS=${SCENARIOS:-1}
if [ "$SCENARIOS" = "1" ]; then
    ROUTES=${ROUTES:-$CARL_WORK_DIR/custom_leaderboard/leaderboard/data/longest6.xml}
else
    ROUTES=${ROUTES:-$PD/pufferlib/ocean/cosim/carla/routes/longest6_no_scenarios.xml}
fi
REPETITIONS=${REPETITIONS:-1}
ROUTE_SUBSET=${ROUTE_SUBSET:-}
PORT_BASE=${PORT_BASE:-2000}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
LOGGING=${LOGGING:-1}
OBS_HTML=${OBS_HTML:-1}
REPORT=${REPORT:-$LOGGING}
EVALUATOR=$CARL_WORK_DIR/original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py
for path in "$CKPT" "$PY" "$CARLA_ROOT/CarlaUE4.sh" "$EVALUATOR" "$ROUTES" "$CARL_WORK_DIR/tools/result_parser.py"; do
    [ -e "$path" ] || { echo "missing $path"; exit 1; }
done
for path in "$PY" "$CARLA_ROOT/CarlaUE4.sh"; do
    [ -x "$path" ] || { echo "not executable: $path (chmod +x, or the filesystem is mounted noexec)"; exit 1; }
done
echo "Evaluating checkpoint: $CKPT"

# GPUs of this allocation; one CARLA server + one evaluator per GPU
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c "^GPU")}
if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt 8 ]; then echo "NUM_GPUS=$NUM_GPUS must be 1..8"; exit 1; fi

# PD first, and drop inherited entries from other PufferDrive checkouts (see 9_nuPlan.sh)
INHERITED_PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v "cosim_Puffer\|/PufferDrive" | paste -sd: -)
export PYTHONPATH="$PD:$CARL_WORK_DIR/original_leaderboard/leaderboard:$CARL_WORK_DIR/original_leaderboard/scenario_runner:$CARLA_ROOT/PythonAPI/carla${INHERITED_PYTHONPATH:+:$INHERITED_PYTHONPATH}"
export SCENARIO_RUNNER_ROOT=$CARL_WORK_DIR/original_leaderboard/scenario_runner
cd "$PD" || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export COSIM_DEVICE=${COSIM_DEVICE:-cpu}
export COSIM_DYNAMICS_SOURCE=pufferdrive
export COSIM_MAX_SPEED_MPS=${COSIM_MAX_SPEED_MPS:-30}
export COSIM_ZERO_PARTNER_STOPPED_TIME=${COSIM_ZERO_PARTNER_STOPPED_TIME:-1}
export COSIM_PEDESTRIAN_MIN_SIZE_M=${COSIM_PEDESTRIAN_MIN_SIZE_M:-0}
export COSIM_OBS_HTML_MAX_STEPS=${COSIM_OBS_HTML_MAX_STEPS:-20000}
export COSIM_OBS_HTML_RENDER=0  # routes save the compact replay only; render_carla_obs_html.py renders all pages into $OUT/obs_html

TAG=longest6$([ "$SCENARIOS" = "1" ] || echo "_noscen")$([ "$REPETITIONS" = "1" ] || echo "_rep$REPETITIONS")
# results live in the model's own eval folder, next to the PufferDrive benchmark evals
OUT=$RUN_DIR/eval/carla_${TAG}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}
mkdir -p "$OUT/routes" "$OUT/aggregate"
{
    echo "checkpoint $CKPT"; echo "routes $ROUTES"; echo "repetitions $REPETITIONS"; echo "gpus $NUM_GPUS"
    echo "git $(git -C "$PD" rev-parse --short HEAD 2>/dev/null) $(git -C "$PD" diff --quiet 2>/dev/null || echo dirty)"
} > "$OUT/run_info.txt"
echo "Results -> $OUT"

mkdir -p "$PD/experiments/logs"  # the build helper keeps its source hash there
PATH="$(dirname "$PY"):$PATH" bash "$PD/scripts/kesai/build_ext_if_changed.sh" "$PD" || exit 1
IMPORTED_AGENT=$("$PY" -c "import pufferlib.ocean.cosim.carla.leaderboard_agent as a; print(a.__file__)") || { echo "ERROR: agent import failed (see traceback above)"; exit 1; }
case "$IMPORTED_AGENT" in
    "$PD"/*) echo "agent: $IMPORTED_AGENT" ;;
    *) echo "ERROR: agent imported from '$IMPORTED_AGENT', expected $PD"; exit 1 ;;
esac

# route ids of the xml, dealt round-robin: gpu w runs ids w, w+N, w+2N, ...
mapfile -t ROUTE_IDS < <("$PY" -c "
import sys, xml.etree.ElementTree as ET
print('\n'.join(r.attrib['id'] for r in ET.parse(sys.argv[1]).getroot().iter('route')))" "$ROUTES")
[ -n "$ROUTE_SUBSET" ] && read -r -a ROUTE_IDS <<< "$ROUTE_SUBSET"
echo "${#ROUTE_IDS[@]} routes over $NUM_GPUS GPUs"

route_done() {  # 0 when the route json holds a record that is not one of the crash states CaRL resubmits
    "$PY" - "$1" <<'EOF'
import json, sys
try:
    records = json.load(open(sys.argv[1]))["_checkpoint"]["records"]
except Exception:
    sys.exit(1)
crashed = ("Failed - Agent couldn't be set up", "Failed", "Failed - Simulation crashed", "Failed - Agent crashed")
sys.exit(0 if records and all(r["status"] not in crashed for r in records) else 1)
EOF
}

run_route() {  # $1 gpu, $2 route id, $3 attempt: one CARLA server + one evaluator run, into $OUT/routes/route_<id>
    local gpu=$1 route_id=$2 attempt=$3
    local route_dir=$OUT/routes/route_$(printf "%02d" "$route_id")
    local port=$((PORT_BASE + gpu * 50)) tm_port=$((PORT_BASE + 6000 + gpu * 50))
    local server_log=$route_dir/carla_server_attempt$attempt.log evaluator_log=$route_dir/evaluator_attempt$attempt.log
    mkdir -p "$route_dir"
    rm -f "$route_dir/result.json"
    # no camera sensor without logging, so the server can skip rendering entirely
    local render_args=(-RenderOffScreen -graphicsadapter="$gpu")
    [ "$LOGGING" = "1" ] || render_args=(-nullrhi)
    "$CARLA_ROOT/CarlaUE4.sh" "${render_args[@]}" -nosound \
        -carla-rpc-port="$port" -carla-streaming-port=$((port + 1)) > "$server_log" 2>&1 &
    local server_pid=$! up=1
    echo "$server_pid" >> "$OUT/server_pids"
    for _ in $(seq 1 60); do
        kill -0 "$server_pid" 2>/dev/null || { echo "[gpu$gpu route$route_id] CARLA server died"; break; }
        "$PY" -c "
import carla, sys
try:
    c = carla.Client('localhost', $port); c.set_timeout(10.0); c.get_world().get_map()
except Exception:
    sys.exit(1)" 2>/dev/null && up=0 && break
        sleep 5
    done
    if [ $up -ne 0 ]; then
        pkill -9 -P "$server_pid" 2>/dev/null; kill -9 "$server_pid" 2>/dev/null
        echo "[gpu$gpu route$route_id] last lines of $server_log:"; tail -n 20 "$server_log"
        return 1
    fi
    local log_env=()
    if [ "$LOGGING" = "1" ]; then
        log_env=(COSIM_TELEMETRY="$route_dir/telemetry" COSIM_WORLD_LOG="$route_dir/world_log" COSIM_DEBUG_CARLA_VIEW="$route_dir/carla_view")
        [ "$OBS_HTML" = "1" ] && log_env+=(COSIM_OBS_HTML="$route_dir/obs_html")
    fi
    env CUDA_VISIBLE_DEVICES="$gpu" "${log_env[@]}" \
        "$PY" -u "$EVALUATOR" --routes "$ROUTES" --routes-subset "$route_id" --repetitions "$REPETITIONS" \
        --agent "$PD/pufferlib/ocean/cosim/carla/leaderboard_agent.py" --agent-config "$CKPT" \
        --checkpoint "$route_dir/result.json" --track MAP --port "$port" --traffic-manager-port "$tm_port" \
        > "$evaluator_log" 2>&1
    # children first: killing the wrapper shell first reparents the UE4 binary and leaves it running
    pkill -9 -P "$server_pid" 2>/dev/null; kill -9 "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null
    route_done "$route_dir/result.json" && return 0
    echo "[gpu$gpu route$route_id] no valid route record; last lines of $evaluator_log:"; tail -n 20 "$evaluator_log"
    return 1
}

run_worker() {  # $1 gpu: its share of the routes, sequentially, each retried up to MAX_ATTEMPTS
    local gpu=$1
    for ((k = gpu; k < ${#ROUTE_IDS[@]}; k += NUM_GPUS)); do
        local route_id=${ROUTE_IDS[k]} attempt
        for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
            echo "[gpu$gpu] route $route_id attempt $attempt $(date +%H:%M:%S)"
            run_route "$gpu" "$route_id" "$attempt" && break
            echo "[gpu$gpu] route $route_id attempt $attempt failed"
        done
    done
}

# only the servers this job started (children first: the UE4 binary outlives its wrapper shell otherwise)
trap 'for p in $(cat "$OUT/server_pids" 2>/dev/null); do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done' EXIT
WORKER_PIDS=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    run_worker "$gpu" > "$OUT/worker_gpu$gpu.log" 2>&1 &
    WORKER_PIDS+=($!)
done
wait "${WORKER_PIDS[@]}"
cat "$OUT"/worker_gpu*.log

# aggregate: CaRL's parser over the per-route jsons -> $OUT/aggregate/results.csv, plus a per-route table
for f in "$OUT"/routes/route_*/result.json; do cp "$f" "$OUT/aggregate/$(basename "$(dirname "$f")").json"; done
"$PY" "$CARL_WORK_DIR/tools/result_parser.py" --xml "$ROUTES" --results "$OUT/aggregate"
"$PY" - "$OUT/aggregate" "${#ROUTE_IDS[@]}" <<'EOF'
import glob, json, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/route_*.json")):
    for r in json.load(open(f))["_checkpoint"]["records"]:
        rows.append((r["route_id"], r["status"], r["scores"]["score_composed"], r["scores"]["score_route"], r["scores"]["score_penalty"]))
print(f"\n{len(rows)} route records of {sys.argv[2]} routes")
for route_id, status, ds, rc, ip in rows:
    print(f"  {route_id:22s} {status:34s} DS {ds:6.2f} RC {rc:6.2f} IP {ip:5.3f}")
if rows:
    n = len(rows)
    print(f"mean DS {sum(r[2] for r in rows)/n:.2f}  RC {sum(r[3] for r in rows)/n:.2f}  IP {sum(r[4] for r in rows)/n:.3f}")
EOF

REPORT_ARGS=()
[ "$OBS_HTML" = "1" ] && REPORT_ARGS+=(--obs-html-dir "$OUT/obs_html")
if [ "$REPORT" = "1" ] && [ "$LOGGING" = "1" ]; then
    "$PY" "$PD/scripts/eval/analyze_carla_cosim.py" "$OUT"/routes/route_* "$OUT/report" "${REPORT_ARGS[@]}" || echo "HTML report failed (results above are unaffected)"
fi
# after the report: the gallery links to it when it exists
if [ "$OBS_HTML" = "1" ] && [ "$LOGGING" = "1" ]; then
    "$PY" "$PD/scripts/eval/render_carla_obs_html.py" "$OUT" --routes "$ROUTES" || echo "obs replay gallery failed (results above are unaffected)"
fi
[ -f "$OUT/report/index.html" ] && echo "report -> $OUT/report/index.html"
[ -f "$OUT/obs_html/index.html" ] && echo "obs replays -> $OUT/obs_html/index.html"
echo "Done -> $OUT"
