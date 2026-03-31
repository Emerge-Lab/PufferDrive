# =============================================================================
# PufferDrive Base Image
# =============================================================================

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV TZ=Europe/Paris \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_DEV=1 \
    PATH="/usr/local/nvidia/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /pufferdrive

# =============================================================================
# System Dependencies
# =============================================================================
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        apt-transport-https \
        ca-certificates \
        gnupg && \
    # Google Cloud SDK
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /tmp/* /var/tmp/*

# =============================================================================
# Install uv + Python + venv
# =============================================================================
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PATH="/root/.local/bin:$PATH"

RUN uv venv --python 3.12 /pufferdrive/.venv

# Activate venv for all subsequent commands
ENV VIRTUAL_ENV=/pufferdrive/.venv \
    PATH="/pufferdrive/.venv/bin:$PATH"

# =============================================================================
# Python Dependencies (heavy packages - cached separately)
# =============================================================================
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        psutil \
        rich \
        rich_argparse \
        pandas \
        tqdm \
        matplotlib \
        imageio \
        pyro-ppl \
        mediapy \
        heavyball \
        neptune \
        wandb \
        tensorboard

# =============================================================================
# PufferLib Installation
# =============================================================================
COPY pyproject.toml setup.py setup.cfg ./
COPY pufferlib ./pufferlib

RUN --mount=type=cache,target=/root/.cache/uv \
    NO_TRAIN=1 NO_OCEAN=1 uv pip install -e .

# Remove .so files in pufferlib/
RUN find /pufferdrive/pufferlib -name "*.so" -delete
