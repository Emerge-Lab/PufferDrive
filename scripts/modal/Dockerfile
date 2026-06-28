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
    # Multi-arch so the same image runs on T4 (sm_75), A100 (sm_80), L4/L40S
    # (sm_89), and H100/H200 (sm_90). Build is ~3-4x slower than single-arch
    # but lets us swap Modal gpu= strings without rebuilding the image.
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.9;9.0"

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

# Install uv — faster, deterministic resolver, replaces pip + venv + the
# constraints-file dance. Modal disallows the Dockerfile ADD directive, so
# fetch via curl.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Project venv at /opt/venv (kept off the overlay/image filesystem in
# spirit — symbolic of the cluster's /scratch/$USER/venvs layout).
RUN uv venv /opt/venv --python 3.12
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV="/opt/venv"

RUN uv pip install setuptools wheel

# torch from pypi is built against CUDA 13; pull the cu128 wheel instead so it
# matches the base image's CUDA toolkit. setup.py's CUDAExtension path checks
# this and refuses to build if torch and host CUDA disagree.
RUN uv pip install --index-url https://download.pytorch.org/whl/cu128 torch

# Pin numpy<2 and pandas<2.2 here so the later `uv pip install -e .` resolve
# can't drag numpy 2 in via pandas 3 and break the C extension's numpy 1 ABI
# (NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION in setup.py). uv's resolver
# respects these caps when computing the install_requires solution, unlike
# pip which would gladly upgrade pre-installed pandas during -e .
RUN uv pip install "numpy<2" "pandas<2.2"

WORKDIR /workspace
