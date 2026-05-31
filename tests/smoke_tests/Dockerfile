# Bit-exact CPU image for the PufferDrive smoke golden test.
#
# Option A: freeze every source of float divergence so the golden metrics in
# data/drive_smoke_golden.json reproduce on ANY AVX2-capable x86-64 host
# (CI runner or dev box). What is pinned:
#   - compiler        : the gcc shipped in python:3.11-slim-bookworm
#   - torch / numpy   : exact CPU wheels (no per-host wheel/kernel skew)
#   - kernel ISA      : oneDNN + MKL/OpenBLAS capped to a fixed ISA / kernel
#   - C sim float math: built baseline-ISA + -ffp-contract=off (see setup.py)
#
# Regenerate the golden (after an intentional pipeline change) INSIDE this image:
#   docker build -f tests/smoke_tests/Dockerfile -t pufferdrive-smoke .
#   docker run --rm -e SMOKE_UPDATE_GOLDEN=1 \
#       -v "$PWD/tests/smoke_tests/data:/app/tests/smoke_tests/data" \
#       pufferdrive-smoke
# then commit tests/smoke_tests/data/drive_smoke_golden.json.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PUFFER_CPU=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_CORETYPE=Haswell \
    MKL_CBWR=COMPATIBLE \
    ONEDNN_MAX_CPU_ISA=AVX2 \
    DNNL_MAX_CPU_ISA=AVX2

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Pin torch/numpy first so the editable install (which imports torch at build
# time) and build_ext both see the exact CPU wheels; --no-build-isolation keeps
# the build from pulling a different default torch.
RUN python -m pip install -U pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0+cpu numpy==2.4.4 \
    && pip install -e . --no-build-isolation --no-cache-dir \
    && python setup.py build_ext --inplace --force

CMD ["python", "-m", "pytest", "-v", "tests/smoke_tests/test_drive_train.py"]
