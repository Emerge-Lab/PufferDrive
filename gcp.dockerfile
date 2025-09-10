# Descr: Image goal is minimum size and to be trained with cloud
ARG BASE_IMAGE_NAME=nvcr.io/nvidia/cuda
ARG BASE_IMAGE_TAG=12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG} AS base

ARG DEBIAN_FRONTEND=noninteractive

FROM base AS builder

# Install all system dependencies required for the build process
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    autoconf libtool flex bison libbz2-dev \
    build-essential htop clang gdb llvm tmux psmisc software-properties-common sudo libglfw3 \
    python3.12-dev ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for the build
ENV READTHEDOCS=True
ENV TORCH_CUDA_ARCH_LIST=Turing

# Install uv, create a virtual environment, and install all Python packages
WORKDIR /puffertank
COPY . ./pufferlib
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && . $HOME/.local/bin/env \
    && uv venv --python 3.12 --prompt 🐡_drive /puffertank/venv \
    && . /puffertank/venv/bin/activate \
    && uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install jax[cuda12] \
    && uv pip install -e ./pufferlib[train] --no-build-isolation \
    && uv pip install glfw==2.7 \
    && git clone https://github.com/pufferai/carbs \
    && uv pip install -e ./carbs

FROM base AS gcp

# Install runtime dependencies and Google Cloud SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglfw3 python3.12 \
    curl apt-transport-https ca-certificates gnupg \
    && curl -sS -L https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && apt-get install -y google-cloud-sdk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment and application code from the builder stage
COPY --from=builder /puffertank/venv /puffertank/venv
WORKDIR /puffertank/pufferlib
COPY --from=builder /puffertank/pufferlib .

# Add the virtual environment's binaries to the PATH
ENV PATH="/puffertank/venv/bin:${PATH}"

# Set the entrypoint for the training job.
RUN chmod +x /puffertank/pufferlib/automation/run_training.sh
ENTRYPOINT ["/puffertank/pufferlib/automation/run_training.sh"]
