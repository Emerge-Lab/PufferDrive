#!/bin/bash
#SBATCH --job-name eval_alpasim
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=400G                       # --gres, --mem and --cpus-per-task are overridable on the sbatch line
#SBATCH --cpus-per-task 96
#SBATCH --output experiments/logs/alpasim_%j.out
#SBATCH --error experiments/logs/alpasim_%j.err
#SBATCH --partition dev
# AlpaSim closed-loop evaluation of a PufferDrive checkpoint on NuRec scenes, the AlpaSim counterpart of 9_nuPlan.sh:
# alpasim-training's teacher driver runs the policy in a shadow env of this checkout (PUFFERDRIVE_ROOT=$PD) on the
# scenes converted to PufferDrive bins; AlpaSim renders, simulates the ego and replays the logged traffic. AlpaSim's
# services run as containers: docker compose on scaleway; on kesai, which has no docker, as enroot containers started
# straight from squashfs images inside this job (CONTAINER_RUNTIME=enroot) or as pyxis job steps (pyxis; every step first
# unpacks its image to node-local disk, ~300 GB per run with this topology).
#
# One-time setup: alpasim-training at ALPAGYM_ROOT (`uv sync --all-packages`), the AlpaSim fork at ALPASIM_ROOT, the NuRec
# scenes as real files at USDZ_DIR, their bins with 130 km/h lane limits at BINS (alpasim-training README step 4) and the
# AlpaSim base image plus the NRE renderer image:
#   docker: the base image archived in ALPASIM_IMAGE_DIR (cd $ALPAGYM_ROOT && ALPASIM_ROOT=$ALPASIM_ROOT
#     OUT_DIR=$ALPASIM_IMAGE_DIR sbatch scripts/build_alpasim_image.sbatch; ~3 h on a dev node, ~31 GB archive), readable
#     by this user; the renderer is pulled from nvcr.io.
#   enroot/pyxis: both as squashfs files in SQSH_DIR, named the way the wizard looks them up (image basename with ':'
#     and '-' replaced by '_'):
#       enroot import -o $SQSH_DIR/nre_ga_26.04.sqsh docker://nvcr.io#nvidia/nre/nre-ga:26.04
#       python scripts/kesai/docker_archive_to_sqsh.py alpasim-base-<version>.tar.zst $SQSH_DIR/alpasim_base_<version>.sqsh
#     the archive being build_alpasim_image.sbatch's, copied over from a docker host (~100 GB of scratch, see the script).
#
# Settings are the plain assignments below. Only RUN_DIR, TOPOLOGY and N_SCENES are also read from the environment: 3_train
# and 7_eval chain this eval after their own by exporting RUN_DIR; a smoke test on a shared node is
#   TOPOLOGY=1gpu N_SCENES=2 sbatch --gres gpu:1 --mem 200G --cpus-per-task 48 scripts/kesai/12_alpasim.sh
# and a full run on two GPUs of a shared node (12 scenes in flight instead of 32) is
#   TOPOLOGY=2gpu sbatch --gres gpu:2 --mem 200G --cpus-per-task 48 scripts/kesai/12_alpasim.sh
set -u
source scripts/kesai/cluster_env.sh || exit 1

# full run: TOPOLOGY=8gpu_64rollouts (whole node) N_SCENES=0 (the whole SCENE_LIST); quick check: see the header
TOPOLOGY=${TOPOLOGY:-8gpu_64rollouts}
N_SCENES=${N_SCENES:-0}
RUN_DIR=${RUN_DIR:-$PD/experiments/k_scaled_0045_1000}
CKPT=$RUN_DIR/final_model.pt
[ -f "$CKPT" ] || CKPT=$(ls "$RUN_DIR"/models/model_*.pt 2>/dev/null | sort | tail -n 1)
GOAL_MODE=gt_time:5  # gt_time:<s> | gt | route
GOAL_RADIUS_M=2      # goal arrival radius Damiano's reference evals use; the teacher's own gt_time default is 6
VRU_SIZE_PAD_M=0.8   # added to the length and width of every pedestrian and two-wheeler box the teacher sees; 0 = true boxes
PARTNER_SPLIT_LENGTH_M=7  # actors longer than this (training max 7 m) reach the teacher as ceil(length / this) touching pieces; 0 = one box
RENDER_VIDEO=0       # one mp4 per scene; 1 also streams the teacher BEV and makes every policy step ~5x slower
OBS_REPLAY=1         # PufferDrive observation replay html per scene -> $OUT/obs_replays
CAM_W=640            # 16:9 render size; the teacher never sees the cameras
CAM_H=360
SIM_TIMEOUT_S=3600   # per batch of PARALLEL scenes
ALPASIM_LOG_LEVEL=WARNING  # AlpaSim services; INFO also keeps the per-event INFO lines runtime workers and renderer print anyway (~110 MB per 400 scenes)
PROMETHEUS=0         # AlpaSim performance telemetry (~80 MB per run); only alpagym_runtime.perf reads it
KEEP_ROLLOUT_LOGS=none  # rollout.asl (~36 MB per scene, offline re-scoring/failure analysis only): all | imperfect (score < 1) | failed | none (never written)
ALPAGYM_ROOT=$HOME/alpasim-training
# held-out split Damiano evaluates on; empty = every converted scene that passed the checks (2149 on the cluster, 1584 of them train)
SCENE_LIST=$ALPAGYM_ROOT/scene_splits/eval_405.txt
export ALPASIM_ROOT=$HOME/alpasim
# shoulder lanes off-road (py123d 1_bernhard_dev)
BINS=$SHARED_DATA/nurec/bins/eval405_130kmh_ndshoulders
case "$CLUSTER" in
    kesai)
        CONTAINER_RUNTIME=enroot  # enroot | pyxis
        USDZ_DIR=$SHARED_DATA/nurec/usdz
        SQSH_DIR=$SHARED_DATA/nurec/sqsh
        FUSE_DIR=$HOME/.cache/alpasim-fuse-tools  # squashfuse + fuse-overlayfs the wizard downloads once for enroot
        ;;
    *)
        CONTAINER_RUNTIME=docker
        # USDZ_DIR stays Damiano's, the shared all-usdzs symlinks dangle inside the containers
        USDZ_DIR=$HOME_BASE/ddacol/code/alpasim-training/data/distill/usdz
        ALPASIM_IMAGE_DIR=$HOME_BASE/ddacol/code/alpasim-training/data/images
        ;;
esac
case "$TOPOLOGY" in
    8gpu*) PARALLEL=32 ;;  # scenes in flight
    2gpu) PARALLEL=12 ;;
    *) PARALLEL=4 ;;
esac
ALPAGYM_PY=$ALPAGYM_ROOT/.venv/bin/python
# uv: the host syncs the AlpaSim checkout's venv before it starts the Wizard
export PATH="$HOME/.local/bin:$PATH"

[ -n "$CKPT" ] || { echo "no checkpoint in $RUN_DIR (final_model.pt or models/model_*.pt)"; exit 1; }
case "$KEEP_ROLLOUT_LOGS" in all|imperfect|failed|none) ;; *) echo "KEEP_ROLLOUT_LOGS must be all, imperfect, failed or none, got $KEEP_ROLLOUT_LOGS"; exit 1 ;; esac
for path in "$CKPT" "$ALPAGYM_PY" "$ALPASIM_ROOT/pyproject.toml" "$BINS/scene_manifest.jsonl" "$USDZ_DIR" ${SCENE_LIST:+"$SCENE_LIST"}; do
    [ -r "$path" ] || { echo "missing or not readable: $path"; exit 1; }
done
[ -f "$(dirname "$CKPT")/config.yaml" ] || [ -f "$(dirname "$(dirname "$CKPT")")/config.yaml" ] \
    || { echo "no config.yaml next to or one level above $CKPT"; exit 1; }
[ -z "$(find "$USDZ_DIR" -type l -print -quit)" ] \
    || { echo "$USDZ_DIR holds symlinks, which dangle inside the renderer container; point USDZ_DIR at the real files"; exit 1; }
case "$CONTAINER_RUNTIME" in
    docker) RUNTIME_TOOLS="docker" ;;
    enroot) RUNTIME_TOOLS="enroot /usr/bin/fusermount" ;;  # fusermount: what the wizard's fusermount3 symlink points at
    pyxis) RUNTIME_TOOLS="srun" ;;
    *) echo "CONTAINER_RUNTIME must be docker, enroot or pyxis, got $CONTAINER_RUNTIME"; exit 1 ;;
esac
for tool in $RUNTIME_TOOLS nvidia-smi uv; do
    command -v "$tool" > /dev/null || { echo "missing tool: $tool"; exit 1; }
done
[ "$CONTAINER_RUNTIME" != pyxis ] || srun --help 2>&1 | grep -q -- --container-image \
    || { echo "srun has no --container-image option: pyxis is not installed on $(hostname)"; exit 1; }
case "$TOPOLOGY" in
    8gpu*)
        NUM_GPUS=$(nvidia-smi -L | grep -c "^GPU")
        [ "$NUM_GPUS" -eq 8 ] || { echo "$TOPOLOGY pins its services to GPUs 0-7 and needs all 8, got $NUM_GPUS"; exit 1; }
        ;;
esac
ALPASIM_VERSION=$(grep -m1 '^version' "$ALPASIM_ROOT/pyproject.toml" | sed -E 's/.*"(.*)".*/\1/')
BASE_TAG=alpasim-base:$ALPASIM_VERSION
NRE_TAG=$(grep -m1 -oE 'nvcr.io/nvidia/nre/[^[:space:]"]+' "$ALPASIM_ROOT/src/wizard/configs/base_config.yaml")
[ -n "$NRE_TAG" ] || { echo "no NRE renderer image in $ALPASIM_ROOT/src/wizard/configs/base_config.yaml"; exit 1; }
if [ "$CONTAINER_RUNTIME" = docker ]; then
    ALPASIM_IMAGE_TAR=$ALPASIM_IMAGE_DIR/alpasim-base-$ALPASIM_VERSION.tar.zst
    [ -r "$ALPASIM_IMAGE_TAR" ] || docker image inspect "$BASE_TAG" > /dev/null 2>&1 \
        || { echo "$ALPASIM_IMAGE_TAR is missing or not readable and $BASE_TAG is not on $(hostname)"; exit 1; }
else
    BASE_SQSH=$SQSH_DIR/$(basename "$BASE_TAG" | sed 's/[:-]/_/g').sqsh
    NRE_SQSH=$SQSH_DIR/$(basename "$NRE_TAG" | sed 's/[:-]/_/g').sqsh
    for sqsh in "$BASE_SQSH" "$NRE_SQSH"; do
        [ -r "$sqsh" ] || { echo "missing image $sqsh; create it as described at the top of this script"; exit 1; }
    done
fi
echo "Evaluating checkpoint: $CKPT"

# drop inherited entries from other PufferDrive checkouts: the teacher would import their pufferlib instead of $PD's
export PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v "cosim_Puffer\|/PufferDrive" | paste -sd: -)
export PUFFERDRIVE_ROOT=$PD
mkdir -p "$PD/experiments/logs"  # the build helper keeps its source hash there
# cp312 binding for the alpasim-training venv; NO_TRAIN skips the training-only CUDA extension
NO_TRAIN=1 PATH="$ALPAGYM_ROOT/.venv/bin:$PATH" bash "$PD/scripts/kesai/build_ext_if_changed.sh" "$PD" || exit 1
IMPORTED_PUFFERLIB=$("$ALPAGYM_PY" -c "import alpagym_distill.teacher, pufferlib; from pufferlib.ocean.drive import binding; print(pufferlib.__file__)") \
    || { echo "ERROR: teacher import failed (see traceback above)"; exit 1; }
case "$IMPORTED_PUFFERLIB" in
    "$PD"/*) echo "pufferlib: $IMPORTED_PUFFERLIB" ;;
    *) echo "ERROR: pufferlib imported from '$IMPORTED_PUFFERLIB', expected $PD"; exit 1 ;;
esac

mapfile -t SCENES < <("$ALPAGYM_PY" - "$BINS/scene_manifest.jsonl" "$N_SCENES" "$SCENE_LIST" <<'EOF'
import json, sys
manifest, n_scenes, scene_list = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = [json.loads(line) for line in open(manifest).read().splitlines()[1:]]
def passed_checks(row):
    if "gt_replay_pass" in row:
        return row["source_parity_pass"] and row["gt_replay_pass"]
    if "offroad_verdict" in row:  # manifests built before alpasim-training d29bdd9 (2026-08-31)
        return row["source_parity_pass"] and row["offroad_verdict"] == "both_pass"
    sys.exit(f"{manifest}: unknown scene row schema, keys: {sorted(row)}")
scenes = [r["alpasim_scene_id"] for r in rows if passed_checks(r)]
print(f"{len(scenes)} of {len(rows)} converted scenes passed the conversion checks", file=sys.stderr)
if scene_list:
    usable = set(scenes)
    listed = [s.strip() for s in open(scene_list) if s.strip() and not s.startswith("#")]
    scenes = [s for s in (s if s.startswith("clipgt-") else "clipgt-" + s for s in listed) if s in usable]
    print(f"{len(scenes)} of {len(listed)} scenes in {scene_list} usable", file=sys.stderr)
sys.stdout.write("".join(s + "\n" for s in (scenes if n_scenes == 0 else scenes[:n_scenes])))
EOF
)
[ ${#SCENES[@]} -gt 0 ] || { echo "no scenes to evaluate"; exit 1; }

# results live in the model's own eval folder, next to the PufferDrive benchmark evals
OUT=$RUN_DIR/eval/alpasim_${GOAL_MODE//:/}_r${GOAL_RADIUS_M}$([ "$VRU_SIZE_PAD_M" = "0" ] || echo "_vrupad$VRU_SIZE_PAD_M")$([ "$PARTNER_SPLIT_LENGTH_M" = "0" ] || echo "_split$PARTNER_SPLIT_LENGTH_M")_$(basename "$CKPT" .pt)_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
{
    echo "checkpoint $CKPT"; echo "scenes ${#SCENES[@]}"; echo "topology $TOPOLOGY parallel $PARALLEL goal_mode $GOAL_MODE goal_radius_m $GOAL_RADIUS_M"
    echo "vru_size_pad_m $VRU_SIZE_PAD_M"
    echo "partner_split_length_m $PARTNER_SPLIT_LENGTH_M"
    echo "container_runtime $CONTAINER_RUNTIME"
    for repo in "$PD" "$ALPAGYM_ROOT" "$ALPASIM_ROOT"; do
        echo "git $repo $(git -C "$repo" rev-parse --short HEAD 2>/dev/null) $(git -C "$repo" diff --quiet 2>/dev/null || echo dirty)"
    done
} > "$OUT/run_info.txt"
echo "Results -> $OUT"

# own port window per job, so evals sharing a node never touch each other's services
BASEPORT=$((6000 + (${SLURM_JOB_ID:-$$} % 90) * 20))
if [ "$CONTAINER_RUNTIME" = docker ]; then
    docker info > /dev/null 2>&1 || { echo "docker is not usable on $(hostname)"; exit 1; }
    export COMPOSE_PROJECT_NAME=alpasim_${SLURM_JOB_ID:-$$}
fi
LOADED_BASE=0
PULLED_NRE=0
cleanup() {
    # the service containers outlive the wizard when the host stops it before they exit
    case "$CONTAINER_RUNTIME" in
        docker)
            local compose_file=$OUT/alpagym/alpasim/wizard_0/docker-compose.yaml
            [ -f "$compose_file" ] && docker compose -f "$compose_file" down --volumes --remove-orphans > /dev/null 2>&1
            for container in $(docker ps -aq --filter "label=com.docker.compose.project=$COMPOSE_PROJECT_NAME"); do
                docker rm -f "$container" > /dev/null 2>&1
            done
            # /var/lib/docker is tmpfs on the compute nodes: an image left behind keeps holding ~40 GB of node RAM
            [ "$LOADED_BASE" = 1 ] && docker rmi "$BASE_TAG" > /dev/null 2>&1
            [ "$PULLED_NRE" = 1 ] && docker rmi "$NRE_TAG" > /dev/null 2>&1
            ;;
        enroot)  # the node is ours alone: every enroot container of this user on it belongs to this eval
            pkill -TERM -u "$(id -un)" -f "enroot start --rw" > /dev/null 2>&1
            ;;
        pyxis)
            squeue -h -s -j "${SLURM_JOB_ID:-0}" -o "%i %j" 2>/dev/null | while read -r step_id step_name; do
                case "$step_name" in alpasim-*) scancel --signal=TERM "$step_id" > /dev/null 2>&1 ;; esac
            done
            ;;
    esac
}
trap cleanup EXIT
case "$CONTAINER_RUNTIME" in
    docker)
        if ! docker image inspect "$BASE_TAG" > /dev/null 2>&1; then
            [ -r "$ALPASIM_IMAGE_TAR" ] || { echo "$BASE_TAG is not on $(hostname) and $ALPASIM_IMAGE_TAR is missing or not readable"; exit 1; }
            echo "Loading $BASE_TAG from $ALPASIM_IMAGE_TAR"
            LOADED_BASE=1
            zstd -dc "$ALPASIM_IMAGE_TAR" | docker load || exit 1
        fi
        if ! docker image inspect "$NRE_TAG" > /dev/null 2>&1; then
            echo "Pulling $NRE_TAG"
            PULLED_NRE=1
            docker pull "$NRE_TAG" || exit 1
        fi
        WIZARD_RUN_METHOD=DOCKER_COMPOSE
        ;;
    enroot)
        echo "enroot $(enroot version 2>&1 | head -n 1) on $(hostname)"
        # the host creates $ALPASIM_ROOT/.venv (uv sync) on the first run; until then the wizard fetches the tools itself
        [ ! -x "$ALPASIM_ROOT/.venv/bin/python" ] \
            || "$ALPASIM_ROOT/.venv/bin/python" -c "from alpasim_wizard.deployment.fuse_tools import ensure_fuse_tools; ensure_fuse_tools('$FUSE_DIR')" \
            || { echo "no squashfuse + fuse-overlayfs in $FUSE_DIR: the wizard downloads them from github.com and archive.ubuntu.com (needs dpkg-deb)"; exit 1; }
        # the wizard's squashfuse is an Ubuntu 18.04 build: nodes without libfuse2 need its libraries copied into $FUSE_DIR/lib
        MISSING_FUSE_LIBS=$([ -x "$FUSE_DIR/bin/squashfuse" ] && LD_LIBRARY_PATH="$FUSE_DIR/lib" ldd "$FUSE_DIR/bin/squashfuse" | grep "not found")
        [ -z "$MISSING_FUSE_LIBS" ] || { echo "$FUSE_DIR/bin/squashfuse cannot run on $(hostname):"; echo "$MISSING_FUSE_LIBS"; \
            echo "apt-get download the packages (libfuse2t64, liblzo2-2), dpkg-deb -x them and copy the .so files into $FUSE_DIR/lib"; exit 1; }
        WIZARD_RUN_METHOD=SLURM_ENROOT
        ;;
    pyxis)
        WIZARD_RUN_METHOD=SLURM
        ;;
esac

# the teacher is an external driver, but the 8gpu topologies re-assert driver.skip=false and abort at startup
[ "$PROMETHEUS" = "1" ] && PROMETHEUS_FLAG=true || PROMETHEUS_FLAG=false
[ "$KEEP_ROLLOUT_LOGS" = "none" ] && SAVE_ROLLOUT_LOG_FLAG=false || SAVE_ROLLOUT_LOG_FLAG=true
OVERRIDES="wizard.baseport=$BASEPORT runtime.endpoints.driver.skip=true wizard.log_level=$ALPASIM_LOG_LEVEL wizard.prometheus.start_prometheus=$PROMETHEUS_FLAG"
OVERRIDES="$OVERRIDES runtime.simulation_config.save_rollout_log=$SAVE_ROLLOUT_LOG_FLAG"
if [ "$CONTAINER_RUNTIME" != docker ]; then
    # root inside the containers as with docker (the image's venv and caches are root's), started in the image's /repo:
    # pyxis would otherwise start the base-image services in the host cwd, which does not exist inside them
    OVERRIDES="$OVERRIDES defines.base_image=$BASE_TAG wizard.sqshcaches=[$SQSH_DIR] wizard.slurm_cpu_bind_none=true"
    for service in renderer physics controller trafficsim runtime prometheus; do
        OVERRIDES="$OVERRIDES services.$service.remap_root=true"
    done
    for service in physics controller trafficsim runtime prometheus; do
        OVERRIDES="$OVERRIDES services.$service.workdir=/repo"
    done
    # enroot's import leaves the NRE entrypoint's $0.runfiles (/app/run.runfiles) empty: name the real Bazel runfiles tree
    NRE_RUNFILES=/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles
    OVERRIDES="$OVERRIDES 'services.renderer.environments=[\"HOME=/tmp\",\"XDG_CACHE_HOME=/tmp/.cache\",\"OMP_NUM_THREADS=1\",\"RUNFILES_DIR=$NRE_RUNFILES\"]'"
    OVERRIDES="$OVERRIDES services.renderer.workdir=$NRE_RUNFILES/_main"
    if [ "$CONTAINER_RUNTIME" = enroot ]; then
        # enroot runs the NRE image's entrypoint (the app itself) in front of the command; /etc/rc sources rc.local first
        printf 'exec "$@"\n' > "$OUT/nre_rc.local"
        OVERRIDES="$OVERRIDES 'services.renderer.volumes=[\"$USDZ_DIR:/mnt/nre-data\",\"$ALPASIM_ROOT/data/nre-artifacts/ego-hoods:/mnt/ego-hoods\",\"$OUT/nre_rc.local:/etc/rc.local\"]'"
        OVERRIDES="$OVERRIDES wizard.fuse_dir=$FUSE_DIR"
    fi
fi
[ "$RENDER_VIDEO" = "1" ] && RENDER_VIDEO_FLAG=true || RENDER_VIDEO_FLAG=false
[ "$OBS_REPLAY" = "1" ] && OBS_REPLAY_FLAG=true || OBS_REPLAY_FLAG=false
echo "${#SCENES[@]} scenes, $TOPOLOGY with $PARALLEL in flight on $CONTAINER_RUNTIME; full log: $OUT/evaluate.log (logs/ once the run is tidied)"
cd "$ALPAGYM_ROOT" || exit 1
"$ALPAGYM_PY" -m alpagym_host.evaluate \
    --model_kind teacher \
    --teacher_checkpoint "$CKPT" \
    --teacher_bins_dir "$BINS" \
    --teacher_goal_mode "$GOAL_MODE" \
    --teacher_goal_radius "$GOAL_RADIUS_M" \
    --teacher_vru_size_pad_m "$VRU_SIZE_PAD_M" \
    --teacher_partner_split_length_m "$PARTNER_SPLIT_LENGTH_M" \
    --render_video "$RENDER_VIDEO_FLAG" \
    --teacher_obs_replay "$OBS_REPLAY_FLAG" \
    --local_scene_dir "$USDZ_DIR" \
    --scene_ids "${SCENES[@]}" \
    --topology "$TOPOLOGY" \
    --wizard_run_method "$WIZARD_RUN_METHOD" \
    --num_parallel_scenes "$PARALLEL" \
    --camera_width "$CAM_W" --camera_height "$CAM_H" \
    --simulation_timeout_s "$SIM_TIMEOUT_S" \
    --run_dir "$OUT/alpagym" \
    --extra_overrides "$OVERRIDES" \
    > "$OUT/evaluate.log" 2>&1
EVAL_STATUS=$?

# results.txt: the official score, AlpaSim's km between incidents (all and at-fault) and the mean of every per-scene metric
if [ -f "$OUT/alpagym/evaluation.yaml" ]; then
    "$ALPAGYM_PY" - "$OUT/alpagym/evaluation.yaml" "$OUT/alpagym/alpasim/wizard_0/aggregate/results-summary.json" "$OUT/results.txt" <<'EOF'
import json, statistics, sys
from pathlib import Path
import yaml
summary = yaml.safe_load(Path(sys.argv[1]).read_text())
aggregate_path = Path(sys.argv[2])
scenes = summary["scenes"]
official = summary.get("official_score")
if official is None:
    lines = [f"official_score n/a  (incomplete: {len(scenes)} scenes finished, {len(summary['failed_scenes'])} failed)"]
else:
    lines = [f"official_score {official['score']:.4f}  scenes {official['num_scored']}  failed {official['num_failed']}"]
rows = []
runs = json.loads(aggregate_path.read_text())["metrics_results"] if aggregate_path.exists() else []
for run in runs:
    prefix = f"{run['run_name']} " if len(runs) > 1 else ""
    for name in ("avg_dist_between_incidents", "avg_dist_between_incidents_at_fault"):
        km_per_incident = run.get(name)
        if km_per_incident is None:
            rows.append((prefix + name, "n/a  (no incidents)"))
            continue
        per_km = f"{1.0 / km_per_incident:.4f} incidents/km" if km_per_incident > 0 else "inf incidents/km"
        rows.append((prefix + name, f"{km_per_incident:9.4f} km  ({per_km})"))
for name in sorted({name for scene in scenes for name in scene["metrics"]}):
    values = [float(scene["metrics"][name]) for scene in scenes if name in scene["metrics"]]
    rows.append((name, f"mean {statistics.mean(values):9.4f}  n {len(values)}"))
width = max((len(name) for name, _ in rows), default=0)
lines += [f"{name:{width}s} {value}" for name, value in rows]
text = "".join(line + "\n" for line in lines)
Path(sys.argv[3]).write_text(text)
sys.stdout.write("\n" + text)
EOF
fi
if [ $EVAL_STATUS -ne 0 ]; then
    echo "AlpaSim evaluation failed (exit $EVAL_STATUS); last host lines of $OUT/evaluate.log:"
    grep -vE '^[a-z0-9_-]+-[0-9]+-[0-9]+ +\|' "$OUT/evaluate.log" | tail -n 30
    for service_log in "$OUT"/alpagym/alpasim/wizard_0/txt-logs/out-*-log.txt; do
        [ -f "$service_log" ] || continue
        echo "last lines of $service_log:"
        tail -n 5 "$service_log"
    done
fi
# the stack is done: take it down while docker-compose.yaml exists, the tidy deletes it
cleanup
# results.txt, obs_replays/, logs/, alpasim/ (+ videos/); container INFO lines stay in evaluate.log for debugging and failed runs
TIDY_FLAGS=""
{ [ "$ALPASIM_LOG_LEVEL" = "INFO" ] || [ $EVAL_STATUS -ne 0 ]; } && TIDY_FLAGS="--keep_container_info"
"$ALPAGYM_PY" "$PD/scripts/eval/tidy_alpasim_run.py" "$OUT" --keep_rollout_logs "$KEEP_ROLLOUT_LOGS" $TIDY_FLAGS || echo "tidy failed"
# score-named pages with the AlpaSim score panel + index.html gallery, as 9_nuPlan.sh and 11_carla_longest6_v2.sh
if [ -d "$OUT/obs_replays" ] && [ -f "$OUT/alpasim/results-summary.json" ]; then
    "$ALPAGYM_PY" "$PD/scripts/eval/render_alpasim_obs_html.py" "$OUT" --prune || echo "obs replay gallery failed"
fi
echo "results -> $OUT/results.txt; obs replays -> $OUT/obs_replays/index.html; logs -> $OUT/logs; per-scene scores + per-step metrics -> $OUT/alpasim"
echo "Done -> $OUT"
exit $EVAL_STATUS
