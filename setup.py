# Builds pufferlib._C from sim/binding.c + src/bindings_cpu.cpp.
# Usage:
#   uv pip install -e .
#   python setup.py build_ext --inplace --force
#
# Env vars:
#   DEBUG=1   debug symbols + sanitizers (Linux only for sanitizers)

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


raylib_dir = download_raylib()
raylib_a = os.path.join(raylib_dir, "lib", "libraylib.a")

extra_compile_args = [
    "-DPLATFORM_DESKTOP",
    "-DENV_NAME=drive",
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
]
extra_link_args = []

if DEBUG:
    extra_compile_args += ["-O0", "-g"]
else:
    extra_compile_args += ["-O2", "-DNDEBUG"]

if SYSTEM == "Darwin":
    omp_prefix = find_libomp_macos()
    extra_compile_args += [
        "-Xpreprocessor",
        "-fopenmp",
        f"-I{os.path.join(omp_prefix, 'include')}",
        # Quiet pre-existing warnings from sim/* flagged when compiled as C++.
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
    include_dirs=[
        numpy.get_include(),
        pybind11.get_include(),
        os.path.join(raylib_dir, "include"),
        "src",
        "sim",
        "vendor",
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    extra_objects=[raylib_a],
)


setup(ext_modules=[drive_ext])
