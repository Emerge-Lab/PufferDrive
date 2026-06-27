# Base image for PufferDrive nightly training on Modal.
#
# Matches the NYU Greene Singularity sif as closely as possible:
#   - CUDA 12.8.1 + cuDNN runtime
#   - Ubuntu 24.04
#   - Python 3.12
#
# The actual repo (and the built .so files) are baked into the image at deploy
# time by scripts/modal/modal_app.py via copy_local_dir + run_commands. This
# Dockerfile only handles the slow, version-stable layer: system libs, Python,
# torch.
#
# Build context is the repo root (not scripts/modal/). Don't reference
# repo files here — modal_app.py copies them in afterward so changes don't
# invalidate the slow base.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Modal A100-80GB is sm_80. Single-arch keeps the C extension build fast.
    TORCH_CUDA_ARCH_LIST=8.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        ccache \
        ca-certificates \
        curl \
        git \
        ninja-build \
        pkg-config \
        # OpenMP for the drive C extension
        libomp-dev \
        # EGL + GL for the headless render path in drive.h (eval.validation_gigaflow)
        libegl1 \
        libegl-dev \
        libgles2-mesa-dev \
        libgl1-mesa-dev \
        libglvnd-dev \
        # Raylib's X11 deps — used by the GLFW fallback when EGL isn't picked
        libx11-dev \
        libxcursor-dev \
        libxinerama-dev \
        libxi-dev \
        libxrandr-dev \
        # Headless display server for the GLFW fallback path
        xvfb \
        # ffmpeg for validation_gigaflow video encoding
        ffmpeg \
        # Python toolchain
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# Heaviest pip deps go in their own layer so iterating on the rest of the
# image doesn't redownload torch (~2GB).
RUN python -m pip install --upgrade --break-system-packages \
        pip \
        setuptools \
        wheel
RUN python -m pip install --break-system-packages \
        "torch>=2.4" \
        "numpy<2"

WORKDIR /workspace
