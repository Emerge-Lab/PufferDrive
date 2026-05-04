#!/bin/bash
# Setup script for PufferDrive Singularity container environment
# This creates an overlay with all dependencies for running on HPC clusters
# with older glibc versions.
#
# Architecture:
#   - The overlay is used ONLY for the miniforge3 base Python interpreter.
#   - All Python packages (torch, pufferlib, etc.) live in a venv on /scratch
#     (regular ext4) instead of the overlay (fuse2fs single-threaded ~10 MB/s).
#     This makes installs/rebuilds ~50x faster than the all-in-overlay approach.
#   - At runtime the venv's bin/python symlinks back to /ext3/miniforge3, which
#     is why we still mount the overlay (read-only) when activating the venv.
#
# Usage:
#   1. Create an overlay (one time): ./setup_container.sh create-overlay
#   2. Install dependencies: sbatch --gres=gpu:1 --wrap "./setup_container.sh install"
#   3. Rebuild C extension: sbatch --gres=gpu:1 --wrap "./setup_container.sh rebuild"

set -e

# Configuration - adjust these paths for your setup (all env-var overridable).
# Defaults match submit_cluster.py --container_overlay / --container_image so
# both scripts agree on which overlay they're reading/writing.
OVERLAY_PATH="${OVERLAY_PATH:-/scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3}"
IMAGE_PATH="${IMAGE_PATH:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
OVERLAY_TEMPLATE="${OVERLAY_TEMPLATE:-/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz}"
CONTAINER_DIR="${CONTAINER_DIR:-$(dirname "$OVERLAY_PATH")}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Venv lives on /scratch (regular ext4) — bypasses fuse2fs entirely for installs.
VENV_PATH="${VENV_PATH:-/scratch/$USER/venvs/pufferdrive}"
# Python from the overlay's miniforge3 (mounted read-only at runtime).
CONTAINER_PYTHON="${CONTAINER_PYTHON:-/ext3/miniforge3/bin/python3}"

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
    mv "${TEMPLATE_NAME%.gz}" overlay.ext3

    echo "Overlay created at $OVERLAY_PATH"
    echo ""
    echo "Next step: Submit the install job:"
    echo "  sbatch --account=YOUR_ACCOUNT --gres=gpu:1 --cpus-per-task=8 --mem=32gb --time=60 \\"
    echo "    --wrap \"$0 install\""
}

# Append the NCCL LD_LIBRARY_PATH fix to a venv's activate script so every
# job that activates the venv inherits it. The cuda12.8 sif ships an old
# libnccl (2.25.1) in /usr/lib that lacks ncclCommShrink; torch 2.x bundles
# its own NCCL — we just have to make sure ld.so finds the bundled one first.
patch_activate_nccl_fix() {
    local activate="$VENV_PATH/bin/activate"
    if grep -q "PUFFERDRIVE_NCCL_FIX" "$activate" 2>/dev/null; then
        return 0
    fi
    cat >> "$activate" <<'EOF'

# PUFFERDRIVE_NCCL_FIX: prepend torch's bundled NCCL so libtorch_cuda finds
# ncclCommShrink (added in NCCL 2.27). Without this, torchrun child procs
# resolve libnccl.so.2 from the sif's old /usr/lib and crash on import torch.
NCCL_DIR=$(compgen -G "$VIRTUAL_ENV/lib/python3.*/site-packages/nvidia/nccl/lib" | head -1)
if [ -n "$NCCL_DIR" ] && [ -d "$NCCL_DIR" ]; then
    export LD_LIBRARY_PATH="$NCCL_DIR:${LD_LIBRARY_PATH:-}"
fi
EOF
}

# Activate (and create if missing) the project venv. Must be called from
# inside the container so $CONTAINER_PYTHON exists.
ensure_venv() {
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo "=== Creating venv at $VENV_PATH ==="
        mkdir -p "$(dirname "$VENV_PATH")"
        "$CONTAINER_PYTHON" -m venv "$VENV_PATH"
    fi
    patch_activate_nccl_fix
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
}

install_deps() {
    echo "=== Installing dependencies into venv at $VENV_PATH ==="
    ensure_venv

    # Multi-arch CUDA build so the _C.so runs on A100 (8.0), L40S (8.9),
    # H100/H200 (9.0) without "no kernel image is available for execution
    # on the device" crashes when a training job lands on a different GPU.
    export TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"
    # Parallel C++/CUDA build (ninja honors MAX_JOBS).
    export MAX_JOBS=8
    # Block ~/.local/lib leakage — venv only.
    export PYTHONNOUSERSITE=1

    # Bootstrap uv into the venv if missing. uv parallelizes wheel extract
    # and is dramatically faster than pip — even more so now that writes
    # go to ext4 instead of fuse2fs.
    if ! command -v uv >/dev/null 2>&1; then
        echo "=== Bootstrapping uv ==="
        pip install --no-cache-dir uv
    fi

    echo "=== Installing build tools (ninja for parallel CUDA compile) ==="
    uv pip install ninja

    # --reinstall heals partial-state venvs cleanly (e.g. after a killed install).
    echo "=== Installing PyTorch ==="
    uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo "=== Installing PufferDrive (editable; also builds C extension via setup.py) ==="
    cd "$PROJECT_ROOT"
    uv pip install -e ".[cluster]"

    echo "=== Installing additional packages ==="
    uv pip install wandb rich submitit pyyaml

    echo "=== Setup complete ==="
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python -c "from pufferlib.ocean.drive import binding; print('C binding loaded successfully')"
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
    python -c "from pufferlib.ocean.drive import binding; print('C binding loaded successfully')"
}

run_in_container() {
    local cmd="$1"
    # Overlay mounted read-only — venv's bin/python symlinks back into
    # /ext3/miniforge3 for the interpreter, but every package read/write
    # happens on /scratch ext4 (the venv on $VENV_PATH).
    singularity exec --nv \
        --overlay "$OVERLAY_PATH:ro" \
        "$IMAGE_PATH" \
        bash -c "cd $PROJECT_ROOT && $cmd"
}

run_in_container_writable() {
    local cmd="$1"
    # --fakeroot still required because uv bootstrap writes to /ext3/miniforge3
    # (the system pip puts uv there before we activate the venv). Once uv
    # is bootstrapped, all subsequent installs go to the venv on /scratch
    # (regular ext4, no fuse2fs in the write path).
    singularity exec --nv --fakeroot \
        --overlay "$OVERLAY_PATH" \
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
            run_in_container_writable "$0 install"
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
        echo "PufferDrive Container Setup"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  create-overlay  Create a new overlay filesystem (run on login node)"
        echo "  install         Install all dependencies into venv on /scratch (submit as GPU job)"
        echo "  rebuild         Rebuild C extension only (submit as GPU job)"
        echo ""
        echo "Environment variables:"
        echo "  VENV_PATH       Where the venv lives (default: /scratch/\$USER/venvs/pufferdrive)"
        echo "  OVERLAY_PATH    Singularity overlay (only needs miniforge3 base python)"
        echo ""
        echo "Example workflow:"
        echo "  1. $0 create-overlay"
        echo "  2. sbatch --gres=gpu:1 --time=60 --wrap \"$0 install\""
        echo "  3. python scripts/submit_cluster.py --container ..."
        ;;
esac
