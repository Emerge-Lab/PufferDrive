"""Pins single-threaded math on a fixed kernel ISA before any smoke test imports numpy or torch."""

import os


SMOKE_THREAD_PINNING = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OPENBLAS_CORETYPE": "Haswell",
    "MKL_CBWR": "COMPATIBLE",
    "ONEDNN_MAX_CPU_ISA": "AVX2",  # oneDNN dispatches by host CPU, which MKL_CBWR does not cover
    "DNNL_MAX_CPU_ISA": "AVX2",
}
for variable_name, value in SMOKE_THREAD_PINNING.items():
    os.environ.setdefault(variable_name, value)
