# Builds pufferlib._C as follows:
#   - macOS / no CUDA: native setuptools build of sim/binding.c +
#     src/bindings_cpu.cpp.
#   - Linux + CUDA: delegates to ./build.sh (which already handles the
#     two-step C/C++/CUDA compile correctly). torch's CUDAExtension can't
#     be used here because it forces every source file through c++,
#     which breaks puffer-4's C-only headers (stdatomic.h, designated
#     initializers, void* implicit conversions).
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
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request

import numpy
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


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
    """Return CUDA_HOME path if nvcc is usable, else None."""
    if FORCE_CPU or SYSTEM != "Linux":
        return None
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    if not os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        return None
    return cuda_home


raylib_dir = download_raylib()
# Absolute paths so ninja's build-subdir CWD doesn't break -I and extra_objects.
raylib_a = os.path.abspath(os.path.join(raylib_dir, "lib", "libraylib.a"))
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

base_compile_args = [
    "-DPLATFORM_DESKTOP",
    "-DENV_NAME=drive",
    # OBS_TENSOR_T must match the typedef in sim/binding.c so pufferlib.cu
    # finds it. build.sh extracts it from the source via awk; we pin it here.
    "-DOBS_TENSOR_T=FloatTensor",
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
]
if DEBUG:
    base_compile_args += ["-O0", "-g"]
else:
    base_compile_args += ["-O2", "-DNDEBUG"]

base_include_dirs = [
    numpy.get_include(),
    pybind11.get_include(),
    os.path.abspath(os.path.join(raylib_dir, "include")),
    os.path.join(PROJECT_ROOT, "src"),
    os.path.join(PROJECT_ROOT, "sim"),
    os.path.join(PROJECT_ROOT, "vendor"),
]

cuda = detect_cuda()
cmdclass = {}

if cuda is not None:
    CUDA_HOME = cuda
    print(f"CUDA detected (CUDA_HOME={CUDA_HOME}); delegating build to ./build.sh")

    class BuildShExt(build_ext):
        """Run ./build.sh, which produces pufferlib/_C${EXT_SUFFIX} directly."""

        def build_extensions(self):
            print(f"=== running {PROJECT_ROOT}/build.sh ===")
            env = os.environ.copy()
            # On the puffer cluster overlays we install conda-forge
            # llvm-openmp, which provides libomp (not libomp5) but isn't on
            # clang's default search path. Detect via sys.prefix (the running
            # python's install root) since CONDA_PREFIX may be unset even
            # inside a miniforge install. Fall back to plain -lomp otherwise.
            if "OMP_LIB" not in env:
                prefix_lib = os.path.join(sys.prefix, "lib")
                if os.path.exists(os.path.join(prefix_lib, "libomp.so")):
                    env["OMP_LIB"] = f"-L{prefix_lib} -lomp"
                else:
                    env["OMP_LIB"] = "-lomp"
            subprocess.check_call(["bash", "build.sh"], cwd=PROJECT_ROOT, env=env)
            # build.sh writes pufferlib/_C${EXT_SUFFIX}. --inplace usage
            # is satisfied. For non-inplace, copy into self.build_lib.
            ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
            so_path = os.path.join(PROJECT_ROOT, "pufferlib", f"_C{ext_suffix}")
            if not os.path.exists(so_path):
                sys.exit(f"build.sh did not produce {so_path}")
            if not self.inplace:
                target = os.path.join(self.build_lib, "pufferlib", f"_C{ext_suffix}")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(so_path, target)
                print(f"copied {so_path} -> {target}")

    # The Extension is just a marker so build_ext is invoked; build.sh handles
    # all the real work and ignores these fields.
    drive_ext = Extension("pufferlib._C", sources=[])
    cmdclass["build_ext"] = BuildShExt
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
