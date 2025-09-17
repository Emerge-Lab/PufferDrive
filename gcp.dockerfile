# Descr: Image goal is to build all the necessary element to run PufferDrive release
ARG BASE_IMAGE_NAME=nvcr.io/nvidia/cuda
ARG BUILD_IMAGE_TAG=12.1.1-cudnn8-devel-ubuntu22.04
ARG RUNTIME_IMAGE_TAG=12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE_NAME}:${BUILD_IMAGE_TAG} AS base

ARG DEBIAN_FRONTEND=noninteractive

FROM base AS builder

# Install all system dependencies required for the build process
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    autoconf libtool flex bison libbz2-dev \
    build-essential htop clang gdb llvm tmux psmisc software-properties-common sudo libglfw3-dev \
    python3.10-dev ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for the build
ENV READTHEDOCS=True
ENV TORCH_CUDA_ARCH_LIST=Turing

# Install uv, create a virtual environment and install all Python packages
WORKDIR /pufferdrive
COPY . .
RUN chmod +x /pufferdrive/automation/run_training.sh

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && . $HOME/.local/bin/env \
    && uv venv --python 3.10 --prompt 🐡_drive \
    && . .venv/bin/activate \
    && uv pip install setuptools wheel cython \
    && uv pip install torch torchvision torchaudio torchdata --index-url https://download.pytorch.org/whl/cu121 \
    && uv pip install jax[cuda12] \
    && uv pip install -e .[train] --no-build-isolation \
    && uv pip install glfw==2.7 \
    && git clone https://github.com/pufferai/carbs \
    && uv pip install -e ./carbs \
    && uv pip install gcsfs

# Descr: Image goal is minimum size and to be trained with cloud computing
FROM ${BASE_IMAGE_NAME}:${RUNTIME_IMAGE_TAG} AS gcp

# Install only essential runtime dependencies, removing the large Google Cloud SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglfw3 python3.10 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory before copying files to ensure correct paths
WORKDIR /pufferdrive

# Copy the virtual environment and application code from the builder stage
COPY --from=builder /pufferdrive/.venv ./.venv
COPY --from=builder /pufferdrive/pufferlib ./pufferlib
COPY --from=builder /pufferdrive/automation ./automation
COPY --from=builder /pufferdrive/resources ./resources
COPY --from=builder /pufferdrive/carbs ./carbs

# Add the virtual environment's binaries to the PATH
ENV PATH="/pufferdrive/.venv/bin:${PATH}"

# Set the entrypoint for the training job.
ENTRYPOINT ["/pufferdrive/automation/run_training.sh"]
