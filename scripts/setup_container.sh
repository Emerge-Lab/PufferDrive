#!/bin/bash
# OpenGigaFlow on HPC clusters with an old glibc: the singularity image supplies CUDA, miniforge3 and the venv live
# on /scratch (no fuse2fs on install paths, the venv activates on the login node too), the overlay is for system tools.
#   1. Create an overlay (one time): ./setup_container.sh create-overlay
#   2. Install dependencies: sbatch --gres=gpu:1 --wrap "./setup_container.sh install"
#   3. Rebuild C extension: sbatch --gres=gpu:1 --wrap "./setup_container.sh rebuild"

set -e

# env-overridable; the defaults match submit_cluster.py's --container_overlay / --container_image
OVERLAY_PATH="${OVERLAY_PATH:-/scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3}"
IMAGE_PATH="${IMAGE_PATH:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
OVERLAY_TEMPLATE="${OVERLAY_TEMPLATE:-/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz}"
CONTAINER_DIR="${CONTAINER_DIR:-$(dirname "$OVERLAY_PATH")}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Venv lives on /scratch (regular ext4) — bypasses fuse2fs entirely for installs.
VENV_PATH="${VENV_PATH:-/scratch/$USER/venvs/pufferdrive}"
# on /scratch too, so the venv's python symlink resolves on any node without the overlay
MINIFORGE3_DIR="${MINIFORGE3_DIR:-/scratch/$USER/miniforge3}"
# a release with Python 3.12: torch's cu121 wheels stop at cp312
MINIFORGE3_INSTALLER_URL="${MINIFORGE3_INSTALLER_URL:-https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh}"
MINIFORGE3_PYTHON_VERSION="${MINIFORGE3_PYTHON_VERSION:-3.12}"
CONTAINER_PYTHON="${CONTAINER_PYTHON:-$MINIFORGE3_DIR/bin/python3}"

create_overlay() {
    echo "=== Creating overlay filesystem ==="
    mkdir -p "$CONTAINER_DIR"

    if [ -f "$OVERLAY_PATH" ]; then
        echo "Overlay already exists at $OVERLAY_PATH"
        echo "Delete it first if you want to recreate: rm $OVERLAY_PATH"
        exit 1
    fi

    echo "Copying and extracting overlay (this may take a few minutes)..."
    cp "$OVERLAY_TEMPLATE" "$CONTAINER_DIR/"
    TEMPLATE_NAME=$(basename "$OVERLAY_TEMPLATE")
    cd "$CONTAINER_DIR"
    gunzip "$TEMPLATE_NAME"

    echo "Overlay created at $OVERLAY_PATH"
    echo ""
    echo "Next step: Submit the install job:"
    echo "  sbatch --account=YOUR_ACCOUNT --gres=gpu:1 --cpus-per-task=8 --mem=32gb --time=60 \\"
    echo "    --wrap \"$0 install\""
}

# every activation puts torch's bundled NCCL first: the sif's libnccl 2.25.1 lacks ncclCommShrink
patch_activate_nccl_fix() {
    local activate="$VENV_PATH/bin/activate"
    if grep -q "OPENGIGAFLOW_NCCL_FIX" "$activate" 2>/dev/null; then
        return 0
    fi
    cat >> "$activate" <<'EOF'

# OPENGIGAFLOW_NCCL_FIX: prepend torch's bundled NCCL so libtorch_cuda finds
# ncclCommShrink (added in NCCL 2.27). Without this, torchrun child procs
# resolve libnccl.so.2 from the sif's old /usr/lib and crash on import torch.
NCCL_DIR=$(compgen -G "$VIRTUAL_ENV/lib/python3.*/site-packages/nvidia/nccl/lib" | head -1)
if [ -n "$NCCL_DIR" ] && [ -d "$NCCL_DIR" ]; then
    export LD_LIBRARY_PATH="$NCCL_DIR:${LD_LIBRARY_PATH:-}"
fi
EOF
}

# miniforge3 on /scratch (plain installer: no root, no singularity), a real python3 on every node
ensure_miniforge3() {
    if [ -x "$MINIFORGE3_DIR/bin/python3" ]; then
        # uv venv would otherwise reuse an earlier install of another Python (e.g. 3.13)
        local existing
        existing="$("$MINIFORGE3_DIR/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
        if [ "$existing" = "$MINIFORGE3_PYTHON_VERSION" ]; then
            return 0
        fi
        echo "=== miniforge3 at $MINIFORGE3_DIR has python $existing (want $MINIFORGE3_PYTHON_VERSION); reinstalling ==="
        rm -rf "$MINIFORGE3_DIR"
    fi
    echo "=== Installing miniforge3 to $MINIFORGE3_DIR ==="
    mkdir -p "$(dirname "$MINIFORGE3_DIR")"
    local installer
    installer="$(mktemp -t miniforge3-installer.XXXXXX.sh)"
    curl -fsSL "$MINIFORGE3_INSTALLER_URL" -o "$installer"
    bash "$installer" -b -p "$MINIFORGE3_DIR"
    rm -f "$installer"
    echo "miniforge3 installed at $MINIFORGE3_DIR"
}

# uv from PATH or a ~/.local/bin (auto-bound by apptainer), else the official installer
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        return 0
    fi
    for cand in "$HOME/.local/bin/uv" "/scratch/$USER/.local/bin/uv"; do
        if [ -x "$cand" ]; then
            UV_BIN="$cand"
            export PATH="$(dirname "$cand"):$PATH"
            return 0
        fi
    done
    echo "=== Bootstrapping uv via installer ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    UV_BIN="$HOME/.local/bin/uv"
}

# inside the container; uv venv because miniforge3's python -m venv has no pip (the system pip would reject installs)
ensure_venv() {
    ensure_uv
    # rebuild a venv whose python resolves outside $MINIFORGE3_DIR (e.g. the old /ext3/miniforge3)
    if [ -f "$VENV_PATH/bin/activate" ]; then
        local resolved
        resolved="$(readlink -f "$VENV_PATH/bin/python" 2>/dev/null || true)"
        case "$resolved" in
            "$MINIFORGE3_DIR"/*) ;;
            *)
                echo "=== Rebuilding stale venv at $VENV_PATH (python points to '$resolved', not under $MINIFORGE3_DIR) ==="
                rm -rf "$VENV_PATH"
                ;;
        esac
    fi
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo "=== Creating venv at $VENV_PATH ==="
        mkdir -p "$(dirname "$VENV_PATH")"
        "$UV_BIN" venv "$VENV_PATH" --python "$CONTAINER_PYTHON" --seed
    fi
    patch_activate_nccl_fix
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
}

install_deps() {
    echo "=== Installing dependencies into venv at $VENV_PATH ==="
    ensure_venv

    # _C.so for A100 (8.0), L40S (8.9) and H100/H200 (9.0): a job may land on any of them
    export TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"
    # Parallel C++/CUDA build (ninja honors MAX_JOBS).
    export MAX_JOBS=8
    # Block ~/.local/lib leakage — venv only.
    export PYTHONNOUSERSITE=1

    echo "=== Installing build tools (ninja for parallel CUDA compile) ==="
    uv pip install ninja

    # --reinstall heals partial-state venvs cleanly (e.g. after a killed install).
    echo "=== Installing PyTorch ==="
    uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo "=== Installing OpenGigaFlow (editable; also builds C extension via setup.py) ==="
    cd "$PROJECT_ROOT"
    # --no-build-isolation: build against the torch above instead of ~6 GB of wheels in a fresh build env
    uv pip install --no-build-isolation -e .

    echo "=== Installing additional packages ==="
    uv pip install wandb rich submitit pyyaml

    echo "=== Setup complete ==="
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python -c "from opengigaflow.ocean.drive import binding; print('C binding loaded successfully')"
}

rebuild_extension() {
    echo "=== Rebuilding C extension ==="
    ensure_venv
    export TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"
    export MAX_JOBS=8
    export PYTHONNOUSERSITE=1
    cd "$PROJECT_ROOT"
    python setup.py build_ext --inplace --force
    echo "=== Rebuild complete ==="
    python -c "from opengigaflow.ocean.drive import binding; print('C binding loaded successfully')"
}

run_in_container() {
    local cmd="$1"
    # overlay read-only: the install and rebuild only touch /scratch (miniforge3 + venv)
    singularity exec --nv \
        --overlay "$OVERLAY_PATH:ro" \
        "$IMAGE_PATH" \
        bash -c "cd $PROJECT_ROOT && $cmd"
}

case "${1:-}" in
    create-overlay)
        create_overlay
        ;;
    install)
        if [ -f /.singularity.d/Singularity ]; then
            install_deps
        else
            # miniforge3 outside the container; uv, pip and build_ext inside it for nvcc and the right glibc
            ensure_miniforge3
            run_in_container "$0 install"
        fi
        ;;
    rebuild)
        if [ -f /.singularity.d/Singularity ]; then
            rebuild_extension
        else
            run_in_container "$0 rebuild"
        fi
        ;;
    *)
        echo "OpenGigaFlow Container Setup"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  create-overlay  Create a new overlay filesystem (run on login node)"
        echo "  install         Install all dependencies into venv on /scratch (submit as GPU job)"
        echo "  rebuild         Rebuild C extension only (submit as GPU job)"
        echo ""
        echo "Environment variables:"
        echo "  MINIFORGE3_DIR  Where the base python lives (default: /scratch/\$USER/miniforge3)"
        echo "  VENV_PATH       Where the venv lives (default: /scratch/\$USER/venvs/pufferdrive)"
        echo "  OVERLAY_PATH    Singularity overlay (kept for system-tool installs; not used by the python flow)"
        echo ""
        echo "Example workflow:"
        echo "  1. $0 create-overlay"
        echo "  2. sbatch --gres=gpu:1 --time=60 --wrap \"$0 install\""
        echo "  3. source \$VENV_PATH/bin/activate && python scripts/submit_cluster.py --container ..."
        ;;
esac
