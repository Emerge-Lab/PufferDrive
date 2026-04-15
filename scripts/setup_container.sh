#!/bin/bash
# Setup script for PufferDrive Singularity container environment
# This creates an overlay with all dependencies for running on HPC clusters
# with older glibc versions.
#
# Usage:
#   1. Create an overlay: ./setup_container.sh create-overlay
#   2. Install dependencies: sbatch --gres=gpu:1 --wrap "./setup_container.sh install"
#   3. Rebuild C extension: sbatch --gres=gpu:1 --wrap "./setup_container.sh rebuild"

set -e

# Configuration - adjust these paths for your setup
CONTAINER_DIR="${CONTAINER_DIR:-/scratch/$USER/containers/pufferdrive}"
OVERLAY_PATH="$CONTAINER_DIR/overlay.ext3"
IMAGE_PATH="${IMAGE_PATH:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
OVERLAY_TEMPLATE="${OVERLAY_TEMPLATE:-/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

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

install_deps() {
    echo "=== Installing dependencies in container ==="

    # Use --break-system-packages since we're in a container with overlay
    export PIP_BREAK_SYSTEM_PACKAGES=1

    # Install PyTorch with CUDA
    echo "=== Installing PyTorch ==="
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Install PufferDrive
    echo "=== Installing PufferDrive ==="
    cd "$PROJECT_ROOT"
    pip3 install -e ".[cluster]"

    # Install additional dependencies
    echo "=== Installing additional packages ==="
    pip3 install wandb rich submitit pyyaml

    # Build the C extension
    echo "=== Building C extension ==="
    python3 setup.py build_ext --inplace --force

    echo "=== Setup complete ==="
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python3 -c "from pufferlib.ocean.drive import binding; print('C binding loaded successfully')"
}

rebuild_extension() {
    echo "=== Rebuilding C extension ==="
    cd "$PROJECT_ROOT"
    python3 setup.py build_ext --inplace --force
    echo "=== Rebuild complete ==="
    python3 -c "from pufferlib.ocean.drive import binding; print('C binding loaded successfully')"
}

run_in_container() {
    local cmd="$1"
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
        # Check if we're already in a container
        if [ -f /.singularity.d/Singularity ]; then
            install_deps
        else
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
        echo "  install         Install all dependencies (submit as GPU job)"
        echo "  rebuild         Rebuild C extension only (submit as GPU job)"
        echo ""
        echo "Example workflow:"
        echo "  1. $0 create-overlay"
        echo "  2. sbatch --gres=gpu:1 --time=60 --wrap \"$0 install\""
        echo "  3. python scripts/submit_cluster.py --container ..."
        ;;
esac
