# Builds pufferlib._C from sim/binding.c plus either:
#   - src/bindings.cu       (CUDA backend, default when CUDA toolkit is found)
#   - src/bindings_cpu.cpp  (CPU fallback)
#
# Usage:
#   uv pip install -e .
#   python setup.py build_ext --inplace --force
#
# Env vars:
#   PUFFER_CPU=1   force CPU build even when CUDA is present
#   DEBUG=1        debug symbols + sanitizers (Linux only for sanitizers)

import os
import platform
import subprocess
import sys
import tarfile
import urllib.request

import numpy
import pybind11
from setuptools import Extension, setup


DEBUG = os.getenv("DEBUG", "0") == "1"
FORCE_CPU = os.getenv("PUFFER_CPU", "0") == "1"
SYSTEM = platform.system()
EXTERNAL_LIB_DIR = "vendor"

RAYLIB_URL = "https://github.com/raysan5/raylib/releases/download/5.5/"
RAYLIB_NAME = "raylib-5.5_macos" if SYSTEM == "Darwin" else "raylib-5.5_linux_amd64"


def download_raylib():
    dest = os.path.join(EXTERNAL_LIB_DIR, RAYLIB_NAME)
    if os.path.exists(dest):
        return dest
    print(f"Downloading {RAYLIB_NAME}...")
    os.makedirs(EXTERNAL_LIB_DIR, exist_ok=True)
    archive_name = f"{RAYLIB_NAME}.tar.gz"
    archive_path = os.path.join(EXTERNAL_LIB_DIR, archive_name)
    urllib.request.urlretrieve(RAYLIB_URL + archive_name, archive_path)
    with tarfile.open(archive_path, "r") as tf:
        if sys.version_info >= (3, 12):
            tf.extractall(EXTERNAL_LIB_DIR, filter="data")
        else:
            tf.extractall(EXTERNAL_LIB_DIR)
    os.remove(archive_path)
    return dest


def find_libomp_macos():
    # libomp isn't shipped with Apple clang; users install via `brew install libomp`.
    # `brew --prefix libomp` returns a path even when the keg is missing, so
    # verify the lib actually exists.
    try:
        prefix = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("libomp not found. Install via: brew install libomp")
    if not os.path.exists(os.path.join(prefix, "lib", "libomp.dylib")):
        sys.exit(f"libomp not installed at {prefix}. Run: brew install libomp")
    return prefix


def detect_cuda():
    """Return (CUDAExtension, BuildExtension, CUDA_HOME) if usable, else None."""
    if FORCE_CPU or SYSTEM != "Linux":
        return None
    try:
        from torch.utils.cpp_extension import (
            CUDA_HOME,
            BuildExtension,
            CUDAExtension,
        )
    except ImportError:
        return None
    if not CUDA_HOME or not os.path.exists(os.path.join(CUDA_HOME, "bin", "nvcc")):
        return None
    return CUDAExtension, BuildExtension, CUDA_HOME


def find_pkg_path(pkg_name):
    """Resolve `import nvidia.<pkg>` to its install dir, or None."""
    try:
        mod = __import__(f"nvidia.{pkg_name}", fromlist=[pkg_name])
        return mod.__path__[0]
    except ImportError:
        return None


def find_cudnn_paths(cuda_home):
    # System install first, then nvidia-cudnn-cu* wheel.
    if os.path.exists(os.path.join(cuda_home, "include", "cudnn.h")):
        return [os.path.join(cuda_home, "include")], [os.path.join(cuda_home, "lib64")]
    if os.path.exists("/usr/include/cudnn.h"):
        return ["/usr/include"], ["/usr/lib/x86_64-linux-gnu"]
    p = find_pkg_path("cudnn")
    if p:
        return [os.path.join(p, "include")], [os.path.join(p, "lib")]
    return [], []


def find_nccl_paths(cuda_home):
    for inc_dir in ("/usr/include", os.path.join(cuda_home, "include")):
        if os.path.exists(os.path.join(inc_dir, "nccl.h")):
            return [inc_dir], [os.path.join(cuda_home, "lib64")]
    p = find_pkg_path("nccl")
    if p:
        return [os.path.join(p, "include")], [os.path.join(p, "lib")]
    return [], []


raylib_dir = download_raylib()
raylib_a = os.path.join(raylib_dir, "lib", "libraylib.a")

base_compile_args = [
    "-DPLATFORM_DESKTOP",
    "-DENV_NAME=drive",
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
]
if DEBUG:
    base_compile_args += ["-O0", "-g"]
else:
    base_compile_args += ["-O2", "-DNDEBUG"]

base_include_dirs = [
    numpy.get_include(),
    pybind11.get_include(),
    os.path.join(raylib_dir, "include"),
    "src",
    "sim",
    "vendor",
]

cuda = detect_cuda()
cmdclass = {}

if cuda is not None:
    CUDAExtension, BuildExtension, CUDA_HOME = cuda
    print(f"Building CUDA extension with CUDA_HOME={CUDA_HOME}")

    cudnn_inc, cudnn_lib = find_cudnn_paths(CUDA_HOME)
    nccl_inc, nccl_lib = find_nccl_paths(CUDA_HOME)

    cxx_flags = base_compile_args + [
        "-fopenmp",
        "-D_GNU_SOURCE",
        "-Wno-implicit-function-declaration",
    ]
    nvcc_flags = base_compile_args + [
        # Match build.sh's defaults; `-DPUFFER_NATIVE_PUFFERL=0` means torch
        # owns the training loop and the C side only steps envs on GPU.
        "-DPUFFER_NATIVE_PUFFERL=0",
        "-DPRECISION_FLOAT",
        "--threads=0",
        "-Xcompiler=-fopenmp",
        "-Xcompiler=-fPIC",
    ]

    drive_ext = CUDAExtension(
        "pufferlib._C",
        sources=["src/bindings.cu", "sim/binding.c"],
        include_dirs=base_include_dirs + cudnn_inc + nccl_inc,
        library_dirs=[os.path.join(CUDA_HOME, "lib64")] + cudnn_lib + nccl_lib,
        libraries=["cudnn", "nccl", "nvidia-ml", "cublas", "cusolver", "curand"],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        extra_link_args=["-fopenmp", "-lGL", "-Bsymbolic-functions"],
        extra_objects=[raylib_a],
    )
    cmdclass["build_ext"] = BuildExtension
else:
    print("Building CPU extension (no CUDA detected, force-cpu, or non-Linux)")

    extra_compile_args = list(base_compile_args)
    extra_link_args = []

    if SYSTEM == "Darwin":
        omp_prefix = find_libomp_macos()
        extra_compile_args += [
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{os.path.join(omp_prefix, 'include')}",
            "-Wno-error=incompatible-function-pointer-types",
            "-Wno-error=incompatible-pointer-types-discards-qualifiers",
        ]
        extra_link_args += [
            f"-L{os.path.join(omp_prefix, 'lib')}",
            "-lomp",
            "-framework", "Cocoa",
            "-framework", "OpenGL",
            "-framework", "IOKit",
        ]
    elif SYSTEM == "Linux":
        extra_compile_args += [
            "-fopenmp",
            "-D_GNU_SOURCE",
            "-Wno-alloc-size-larger-than",
            "-Wno-implicit-function-declaration",
        ]
        extra_link_args += ["-fopenmp", "-lGL", "-Bsymbolic-functions"]
        if DEBUG:
            sanitize = ["-fsanitize=address,undefined,bounds,pointer-overflow,leak", "-fno-omit-frame-pointer"]
            extra_compile_args += sanitize
            extra_link_args += sanitize
    else:
        sys.exit(f"Unsupported system: {SYSTEM}")

    drive_ext = Extension(
        "pufferlib._C",
        sources=["src/bindings_cpu.cpp", "sim/binding.c"],
        include_dirs=base_include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        extra_objects=[raylib_a],
    )


setup(ext_modules=[drive_ext], cmdclass=cmdclass)
