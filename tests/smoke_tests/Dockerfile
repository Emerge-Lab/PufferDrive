# Pinned CPU image of the smoke goldens (gcc, wheels, BLAS kernels, QEMU CPU); regenerate them with update_goldens.sh
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_CORETYPE=Haswell \
    MKL_CBWR=COMPATIBLE \
    ONEDNN_MAX_CPU_ISA=AVX2 \
    DNNL_MAX_CPU_ISA=AVX2

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential qemu-user-static \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# the exact CPU wheels first: the editable install imports torch at build time, without build isolation
RUN python -m pip install -U pip pytest \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0+cpu numpy==2.4.4 \
    && pip install -e . --no-build-isolation --no-cache-dir \
    && python setup.py build_ext --inplace --force

# QEMU shows every host the same Haswell CPUID, so libm and torch pick the same kernels; CMD is the default test list
ENTRYPOINT ["qemu-x86_64-static", "-cpu", "Haswell", "/usr/local/bin/python", "-m", "pytest", "-v"]
CMD ["tests/smoke_tests/test_drive_train.py", "tests/smoke_tests/test_drive_rollout.py"]
