#!/bin/bash
# Setup script for PufferDrive Singularity container environment
# This creates an overlay with all dependencies for running on HPC clusters
# with older glibc versions.
#
# Architecture:
#   - miniforge3 lives on /scratch (NOT in the overlay) so its python is a
#     real file accessible from any node, in or out of singularity. The venv
#     symlinks `bin/python` into the /scratch miniforge3, which makes
#     `source venv/activate` work on the login node directly without
#     needing to enter the container.
#   - All Python packages (torch, pufferlib, etc.) live in the venv on /scratch
#     too — fuse2fs is not on the write path for any install step.
#   - The singularity image still supplies CUDA + cuDNN at job runtime. The
#     overlay is preserved for the rare case where you need to install
#     system-level tools, but it's not used for the standard python flow.
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
# miniforge3 lives on /scratch too so the venv's python symlink resolves
# from any node without needing the singularity overlay to be mounted.
MINIFORGE3_DIR="${MINIFORGE3_DIR:-/scratch/$USER/miniforge3}"
# Pin to a miniforge3 release that ships Python 3.12. 25.x switched to 3.13,
# but torch's cu121 wheels are cp39..cp312 only (no cp313), so 3.13 breaks
# the install. Bump this once torch publishes cp313 wheels for our index.
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

    if [ ! -f "$OVERLAY_TEMPLATE" ]; then
        echo "ERROR: overlay template not found at $OVERLAY_TEMPLATE"
        echo "The default template path exists only on NYU Greene. Either:"
        echo "  - export OVERLAY_TEMPLATE to your cluster's gzipped blank-ext3 template, or"
        echo "  - create the overlay directly (singularity >= 3.8: 'singularity overlay create'):"
        echo "      apptainer overlay create --size 15360 --create-dir /ext3 \"$OVERLAY_PATH\""
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

# Install miniforge3 to /scratch if it isn't there yet. The conda-forge
# installer is a self-contained shell script — no root, no singularity
# required. Doing this on /scratch (rather than inside the overlay)
# means $MINIFORGE3_DIR/bin/python3 is a real file accessible from any
# node, so the venv's bin/python symlink resolves outside singularity too.
ensure_miniforge3() {
    if [ -x "$MINIFORGE3_DIR/bin/python3" ]; then
        # Verify the existing miniforge3 has the python version we expect —
        # otherwise an earlier install that grabbed "latest" (Python 3.13)
        # would stay around, and uv venv would happily reuse it.
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

# Find or bootstrap a uv binary. Prefer one already on PATH or in
# $HOME/.local/bin (auto-bound by apptainer). Fall back to the official
# installer, which drops a static binary into ~/.local/bin in seconds.
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

# Activate (and create if missing) the project venv. Must be called from
# inside the container so $CONTAINER_PYTHON exists. We use `uv venv` rather
# than `python -m venv` because miniforge3's `python -m venv` produces a
# venv without bin/pip (its ensurepip path is broken on conda envs), which
# causes subsequent pip calls to fall through to Ubuntu 24.04's system pip
# (PEP 668 externally-managed → install rejected). uv venv ships pip out
# of the box and works against any cpython.
ensure_venv() {
    ensure_uv
    # If the venv exists but its python doesn't resolve into the current
    # $MINIFORGE3_DIR (e.g. it points at /ext3/miniforge3 from before we
    # moved miniforge3 onto /scratch), rebuild. readlink -f resolves the
    # whole symlink chain, so this catches the case where the link is
    # valid inside the container (overlay mounted) but stale relative to
    # where the new venv should point.
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

    # Multi-arch CUDA build so the _C.so runs on A100 (8.0), L40S (8.9),
    # H100/H200 (9.0) without "no kernel image is available for execution
    # on the device" crashes when a training job lands on a different GPU.
    export TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"
    # Parallel C++/CUDA build (ninja honors MAX_JOBS).
    export MAX_JOBS=8
    # Block ~/.local/lib leakage — venv only.
    export PYTHONNOUSERSITE=1

    # uv was bootstrapped by ensure_venv; reuse the same binary.

    echo "=== Installing build tools (ninja for parallel CUDA compile) ==="
    uv pip install ninja

    # --reinstall heals partial-state venvs cleanly (e.g. after a killed install).
    echo "=== Installing PyTorch ==="
    uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo "=== Installing PufferDrive (editable; also builds C extension via setup.py) ==="
    cd "$PROJECT_ROOT"
    # --no-build-isolation: pyproject.toml's [build-system].requires includes
    # torch, which would otherwise make uv spin up a fresh build env and
    # download/extract ~6 GB of torch+CUDA wheels just to run setup.py once.
    # We just installed torch into the venv, so use it directly.
    uv pip install --no-build-isolation -e ".[cluster]"

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
    # Overlay mounted read-only — every read/write the install or rebuild
    # cares about happens on /scratch ext4 (miniforge3 + venv). The overlay
    # is kept on the mount line for backward compatibility, but nothing
    # in the python flow writes to it.
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
            # miniforge3 installs on /scratch via plain shell — no singularity
            # needed for that step. The rest (uv + pip + build_ext) runs in
            # the container so nvcc and the right glibc are on PATH.
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
