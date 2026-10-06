#!/bin/bash
# Bootstrap a standalone GPU node (e.g. a vast.ai box) for SHIFT training:
# clone the branch, build the extension for the node's GPUs, fetch the nuPlan
# eval set, and smoke-test the sim. Idempotent; rerun after a failure.
#
#   bash setup_standalone_node.sh            # on the node, as root or with sudo
#
# Overridable via the environment:
#   REPO_DIR   checkout location (default /workspace/PufferDrive)
#   BRANCH     branch to check out (default SHIFT)
#   EVAL_SET   dataset name from data_utils/datasets.yaml linked in as the nuPlan
#              eval maps (default nuplan_mini_val)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/PufferDrive}"
BRANCH="${BRANCH:-SHIFT}"
EVAL_SET="${EVAL_SET:-nuplan_mini_val}"
REPO_URL="https://github.com/Emerge-Lab/PufferDrive.git"

echo "== packages"
if command -v apt-get >/dev/null; then
    apt-get update -qq && apt-get install -y -qq git build-essential tmux curl > /dev/null
fi
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "== checkout $BRANCH into $REPO_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin "$BRANCH" && git checkout "$BRANCH" && git reset --hard "origin/$BRANCH"

echo "== venv"
[ -d .venv ] || uv venv
# shellcheck disable=SC1091
source .venv/bin/activate
# torch is a build requirement; install a CUDA 12.8 wheel first (Blackwell needs sm_120) and
# build the package against it instead of an isolated PyPI torch.
uv pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128 > /dev/null
uv pip install "numpy<2.0" setuptools wheel > /dev/null
uv pip install -e . --no-build-isolation > /dev/null

echo "== build extension for this node's GPUs"
COMPUTE_CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -)"
export TORCH_CUDA_ARCH_LIST="$COMPUTE_CAPS"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
python setup.py build_ext --inplace --force > build.log 2>&1 || { tail -30 build.log; exit 1; }
grep -i "warning:" build.log || true

echo "== nuPlan eval maps ($EVAL_SET)"
python data_utils/fetch_data.py "$EVAL_SET"
NUPLAN_LINK="pufferlib/resources/drive/binaries/nuplan"
if [ ! -L "$NUPLAN_LINK" ]; then
    mv "$NUPLAN_LINK" "${NUPLAN_LINK}_shipped"
    ln -s "$REPO_DIR/data/$EVAL_SET" "$NUPLAN_LINK"
fi
echo "nuplan bins available: $(ls "$NUPLAN_LINK" | grep -c '\.bin$')"

echo "== smoke test"
python - <<'PY'
import numpy as np, torch
from pufferlib.ocean.drive.drive import Drive
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "gpus")
env = Drive(num_agents=64, min_agents_per_env=8, max_agents_per_env=32, num_maps=2,
            map_dir="pufferlib/resources/drive/binaries/carla", simulation_mode="gigaflow",
            scenario_length=64, jerk_rear_axle_slip=True, use_map_cache=True)
env.reset(seed=0)
for _ in range(32):
    env.step(np.zeros_like(env.actions))
env.close()
x = torch.randn(4096, 4096, device="cuda") @ torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
print("sim and cuda ok")
PY
echo "== done. Launch with: cd $REPO_DIR && source .venv/bin/activate && ./scripts/launch_shift_pretrain_node.sh"
