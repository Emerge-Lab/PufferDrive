# Debug command:
#    DEBUG=1 python setup.py build_ext --inplace --force

from setuptools import find_packages, find_namespace_packages, setup, Extension
import numpy
import os
import urllib.request
import zipfile
import tarfile
import platform
import shutil
import sys

from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension

DEBUG = os.getenv("DEBUG", "0") == "1"
NO_OCEAN = os.getenv("NO_OCEAN", "0") == "1"
NO_TRAIN = os.getenv("NO_TRAIN", "0") == "1"

EXTERNAL_LIB_DIR = "extern"
os.makedirs(EXTERNAL_LIB_DIR, exist_ok=True)

RAYLIB_URL = "https://github.com/raysan5/raylib/releases/download/5.5/"
RAYLIB_NAME = "raylib-5.5_macos" if platform.system() == "Darwin" else "raylib-5.5_linux_amd64"
RLIGHTS_URL = "https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/examples/shaders/rlights.h"
INIH_URL = "https://github.com/benhoyt/inih/archive/refs/tags/{tag}.{ext}"


def download_raylib(name, ext):
    dest = os.path.join(EXTERNAL_LIB_DIR, name)
    if os.path.exists(dest):
        return
    print(f"Downloading Raylib {name}")
    archive = name + ext
    urllib.request.urlretrieve(RAYLIB_URL + archive, archive)
    if ext == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(EXTERNAL_LIB_DIR)
    else:
        with tarfile.open(archive, "r") as tf:
            tf.extractall(EXTERNAL_LIB_DIR, filter="data") if sys.version_info >= (3, 12) else tf.extractall(
                EXTERNAL_LIB_DIR
            )
    os.remove(archive)
    urllib.request.urlretrieve(RLIGHTS_URL, os.path.join(dest, "include", "rlights.h"))


def download_inih():
    dest = os.path.join(EXTERNAL_LIB_DIR, "inih-r62")
    if os.path.exists(dest):
        return
    print("Downloading inih")
    url = INIH_URL.format(tag="r62", ext="tar.gz")
    archive = "inih-r62.tar.gz"
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive, "r") as tf:
        members = [m for m in tf.getmembers() if os.path.basename(m.name) in ["ini.c", "ini.h"]]
        tf.extractall(EXTERNAL_LIB_DIR, members=members, filter="data") if sys.version_info >= (
            3,
            12,
        ) else tf.extractall(EXTERNAL_LIB_DIR, members=members)
    os.remove(archive)


if not NO_OCEAN:
    download_inih()


# Shared compile args for all platforms
extra_compile_args = [
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    "-DPLATFORM_DESKTOP",
]
extra_link_args = ["-fwrapv"]
cxx_args = [
    "-fdiagnostics-color=always",
]
nvcc_args = []

if DEBUG:
    extra_compile_args += [
        "-O0",
        "-g",
        "-fsanitize=address,undefined,bounds,pointer-overflow,leak",
        "-fno-omit-frame-pointer",
    ]
    extra_link_args += [
        "-g",
        "-fsanitize=address,undefined,bounds,pointer-overflow,leak",
    ]
    cxx_args += [
        "-O0",
        "-g",
    ]
    nvcc_args += [
        "-O0",
        "-g",
    ]
else:
    extra_compile_args += [
        "-O2",
        "-flto",
    ]
    extra_link_args += [
        "-O2",
    ]
    cxx_args += [
        "-O3",
    ]
    nvcc_args += [
        "-O3",
    ]

system = platform.system()
if system == "Linux":
    extra_compile_args += [
        "-Wno-alloc-size-larger-than",
        "-Wno-implicit-function-declaration",
        "-fmax-errors=3",
    ]
    extra_link_args += [
        "-Bsymbolic-functions",
    ]
    if not NO_OCEAN:
        download_raylib(RAYLIB_NAME, ".tar.gz")
elif system == "Darwin":
    extra_compile_args += [
        "-Wno-error=int-conversion",
        "-Wno-error=incompatible-function-pointer-types",
        "-Wno-error=implicit-function-declaration",
    ]
    extra_link_args += [
        "-framework",
        "Cocoa",
        "-framework",
        "OpenGL",
        "-framework",
        "IOKit",
    ]
    if not NO_OCEAN:
        download_raylib(RAYLIB_NAME, ".tar.gz")
else:
    raise ValueError(f"Unsupported system: {system}")


class BuildExt(build_ext):
    def run(self):
        # Propagate any build_ext options (e.g., --inplace, --force) to subcommands
        build_ext_opts = self.distribution.command_options.get("build_ext", {})
        if build_ext_opts:
            # Copy flags so build_torch and build_c respect inplace/force
            self.distribution.command_options["build_torch"] = build_ext_opts.copy()
            self.distribution.command_options["build_c"] = build_ext_opts.copy()

        # Run the torch and C builds (which will handle copying when inplace is set)
        self.run_command("build_torch")
        self.run_command("build_c")


class CBuildExt(build_ext):
    def run(self, *args, **kwargs):
        self.extensions = [e for e in self.extensions if e.name != "pufferlib._C"]
        super().run(*args, **kwargs)


class TorchBuildExt(cpp_extension.BuildExtension):
    def run(self):
        self.extensions = [e for e in self.extensions if e.name == "pufferlib._C"]
        super().run()


RAYLIB_DIR = os.path.join(EXTERNAL_LIB_DIR, RAYLIB_NAME)
RAYLIB_A = os.path.join(RAYLIB_DIR, "lib", "libraylib.a")
INIH_DIR = os.path.join(EXTERNAL_LIB_DIR, "inih-r62")

c_extensions = []
c_extension_paths = []
if not NO_OCEAN:
    c_extension_paths = ["pufferlib/ocean/drive"]
    c_extensions = [
        Extension(
            "pufferlib.ocean.drive.binding",
            sources=["pufferlib/ocean/drive/binding.c", os.path.join(INIH_DIR, "ini.c")],
            include_dirs=[numpy.get_include(), os.path.join(RAYLIB_DIR, "include")],
            extra_compile_args=extra_compile_args
            + [
                '-DINI_START_COMMENT_PREFIXES="#"',
                '-DINI_INLINE_COMMENT_PREFIXES="#"',
            ],
            extra_link_args=extra_link_args,
            extra_objects=[RAYLIB_A],
        )
    ]


# Check if CUDA compiler is available. You need cuda dev, not just runtime.
torch_extensions = []
if not NO_TRAIN:
    torch_sources = [
        "pufferlib/extensions/pufferlib.cpp",
    ]
    if shutil.which("nvcc"):
        extension = CUDAExtension
        torch_sources.append("pufferlib/extensions/cuda/pufferlib.cu")
    else:
        extension = CppExtension

    torch_extensions = [
        extension(
            "pufferlib._C",
            torch_sources,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        ),
    ]

# Prevent Conda from injecting garbage compile flags
from distutils.sysconfig import get_config_vars  # noqa: E402

cfg_vars = get_config_vars()
for key in ("CC", "CXX", "LDSHARED"):
    if cfg_vars[key]:
        cfg_vars[key] = cfg_vars[key].replace("-B /root/anaconda3/compiler_compat", "")
        cfg_vars[key] = cfg_vars[key].replace("-pthread", "")
        cfg_vars[key] = cfg_vars[key].replace("-fno-strict-overflow", "")

for key, value in cfg_vars.items():
    if value and "-fno-strict-overflow" in str(value):
        cfg_vars[key] = value.replace("-fno-strict-overflow", "")

install_requires = [
    "setuptools",
    "numpy<2.0",
    "gymnasium==0.29.1",
]

if not NO_TRAIN:
    install_requires += [
        "torch",
        "psutil",
        "rich",
        "rich_argparse",
        "pandas",
        "tqdm",
        "matplotlib",
        "imageio",
        "mediapy",
        "heavyball<2.0.0",
        "neptune",
        "wandb",
    ]

setup(
    version="3.0.0",
    packages=find_namespace_packages() + find_packages() + c_extension_paths + ["pufferlib/extensions"],
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=c_extensions + torch_extensions,
    cmdclass={
        "build_ext": BuildExt,
        "build_torch": TorchBuildExt,
        "build_c": CBuildExt,
    },
    include_dirs=[numpy.get_include(), os.path.join(RAYLIB_DIR, "include")],
)
