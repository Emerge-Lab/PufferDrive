# =============================================================================
# PufferDrive Training Image
# =============================================================================

FROM europe-west4-docker.pkg.dev/valeo-cp2386-dev/build-images/pufferdrive_cuda12.8_py3.12:latest

# =============================================================================
# Build C Extensions (only C files to maximize cache hits)
# =============================================================================
COPY pufferlib/ocean ./pufferlib/ocean
COPY pufferlib/extensions/ ./pufferlib/extensions/

ARG GPU_ARCH
ENV TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
RUN python setup.py build_ext --inplace --force

# =============================================================================
# Copy Source Code
# =============================================================================
COPY --chmod=755 mlops/run_training.sh /pufferdrive/run_training.sh
COPY pufferlib/*.py ./pufferlib/
COPY pufferlib/resources ./pufferlib/resources
COPY pufferlib/config ./pufferlib/config

# =============================================================================
# Copy Staged Model (For Finetuning)
# =============================================================================
COPY mlops/staging_models/ /pufferdrive/preload_model/

# =============================================================================
# Entrypoint
# =============================================================================
ENTRYPOINT ["/pufferdrive/run_training.sh"]
