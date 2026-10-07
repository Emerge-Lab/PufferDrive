#!/bin/bash
# nuPlan co-sim env for pufferlib/ocean/cosim/nuplan: Python 3.10 venv with the nuplan-devkit runtime deps,
# CaRL's carl_nuplan, and a separate PufferDrive checkout. The checkout must be separate: build_ext --inplace
# links the extension against this venv's torch and would break the training venv sharing the tree.
# The devkit pins Python 3.9-era packages (hydra 1.1 fails on 3.11, opencv 4.5.1 has no wheels past 3.9)
# while pufferlib pins packages needing 3.11, so pufferlib is installed with --no-deps.
set -u
WS=${WS:-/workspace/carl_workspace}; VENV=${VENV:-/workspace/cosim_venv310}; PD=${PD:-/workspace/PufferDrive_cosim}
mkdir -p "$WS"
[ -d "$WS/nuplan-devkit" ] || git clone -q --depth 1 https://github.com/motional/nuplan-devkit.git "$WS/nuplan-devkit"
[ -d "$WS/CaRL" ] || git clone -q --depth 1 https://github.com/autonomousvision/CaRL.git "$WS/CaRL"
[ -d "$PD/.git" ] || git clone -q --branch SHIFT https://github.com/Emerge-Lab/PufferDrive.git "$PD"
cd "$PD" && git fetch -q origin SHIFT && git reset -q --hard origin/SHIFT
[ -d "$VENV" ] || uv venv -q --python 3.10 "$VENV"
source "$VENV/bin/activate"
uv pip install -q "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "numpy==1.23.4" "setuptools<70" wheel "cython<3"
SKIP="guppy3|grpcio|docker|moto|mock|coverage|pre-commit|hypothesis|testbook|jupyter|selenium|s3fs|setuptools|torch|opencv|numpy"
grep -vE "^\s*#|^\s*$" "$WS/nuplan-devkit/requirements.txt" | sed 's/#.*//' | grep -vE "$SKIP" | while read -r pkg; do
    uv pip install -q --no-build-isolation "$pkg" "numpy==1.23.4" > /dev/null 2>&1 && echo "ok   $pkg" || echo "FAIL $pkg"
done
for pkg in "opencv-python-headless<4.9" "pytorch-lightning==1.9.5" "torchmetrics<1.0" tensorboard "gymnasium==0.26.3" "scikit-learn" jsonpickle tensorboardX pytictoc ujson pyyaml rich psutil "omegaconf"; do
    uv pip install -q --no-build-isolation "$pkg" "numpy==1.23.4" "torch==2.8.0" --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu128 > /dev/null 2>&1 && echo "ok   $pkg" || echo "FAIL $pkg"
done
uv pip install -q --no-deps --no-build-isolation -e "$WS/nuplan-devkit" && echo "ok   nuplan-devkit"
uv pip install -q --no-deps --no-build-isolation -e "$WS/CaRL/nuPlan" && echo "ok   carl"
uv pip install -q "setuptools>=64" > /dev/null
cd "$PD" && uv pip install -q --no-deps --no-build-isolation -e . && echo "ok   pufferlib (no deps)"
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -)"
python setup.py build_ext --inplace --force > "$WS/pd_build_cosim.log" 2>&1 && echo "ok   pufferlib extension" || tail -20 "$WS/pd_build_cosim.log"
python - <<'PY'
import importlib
for mod in ["torch", "cv2", "hydra", "nuplan.planning.simulation.planner.abstract_planner",
            "nuplan.planning.script.run_simulation", "carl_nuplan", "pufferlib.ocean.drive.binding",
            "pufferlib.ocean.cosim.nuplan.planner"]:
    try:
        importlib.import_module(mod); print("import ok  ", mod)
    except Exception as e:
        print("import FAIL", mod, "->", type(e).__name__, str(e)[:220])
PY
uv cache clean -q
echo COSIM_ENV_DONE
