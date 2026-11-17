# DEBUG=1 python setup.py build_ext --inplace --force: -O0 sanitizer build
# NO_TRAIN=1: no torch extension and no training dependencies

from setuptools import find_namespace_packages, setup, Extension
import numpy
import os
import platform

from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME, ROCM_HOME

# CUDA kernel only when torch finds a CUDA / ROCm toolkit; may need `uv pip install --no-build-isolation`
BUILD_CUDA_EXT = bool(CUDA_HOME or ROCM_HOME)

DEBUG = os.getenv("DEBUG", "0") == "1"
NO_TRAIN = os.getenv("NO_TRAIN", "0") == "1"

extra_compile_args = [
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
]
extra_link_args = ["-fwrapv"]
cxx_args = [
    "-fdiagnostics-color=always",
    # no FMA contraction: the advantage kernel rounds the same on every CPU
    "-ffp-contract=off",
]
nvcc_args = []

if DEBUG:
    # Apple clang has no LeakSanitizer; -fsanitize=leak is a hard error there.
    if platform.system() == "Darwin":
        sanitize_flag = "-fsanitize=address,undefined,bounds,pointer-overflow"
    else:
        sanitize_flag = "-fsanitize=address,undefined,bounds,pointer-overflow,leak"
    extra_compile_args += [
        "-O0",
        "-g",
        sanitize_flag,
        "-fno-omit-frame-pointer",
    ]
    extra_link_args += [
        "-g",
        sanitize_flag,
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
        "-O3",
        "-flto=auto",
    ]
    extra_link_args += [
        "-O3",
        "-flto=auto",
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
        # no FMA contraction (and never -march=native): the smoke golden stays bit-identical across machines
        "-ffp-contract=off",
    ]
    extra_link_args += [
        "-Bsymbolic-functions",
    ]
elif system == "Darwin":
    extra_compile_args += [
        "-Wno-error=int-conversion",
        "-Wno-error=incompatible-function-pointer-types",
        "-Wno-error=implicit-function-declaration",
    ]
else:
    raise ValueError(f"Unsupported system: {system}")


class BuildExt(build_ext):
    def run(self):
        # build_torch and build_c inherit --inplace / --force
        build_ext_opts = self.distribution.command_options.get("build_ext", {})
        if build_ext_opts:
            self.distribution.command_options["build_torch"] = build_ext_opts.copy()
            self.distribution.command_options["build_c"] = build_ext_opts.copy()

        self.run_command("build_torch")
        self.run_command("build_c")


class CBuildExt(build_ext):
    def run(self, *args, **kwargs):
        self.extensions = [e for e in self.extensions if e.name != "opengigaflow._C"]
        super().run(*args, **kwargs)


class TorchBuildExt(cpp_extension.BuildExtension):
    def run(self):
        self.extensions = [e for e in self.extensions if e.name == "opengigaflow._C"]
        super().run()


c_extension_paths = ["opengigaflow/ocean/drive"]
c_extensions = [
    Extension(
        "opengigaflow.ocean.drive.binding",
        sources=["opengigaflow/ocean/drive/binding.c"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]


torch_extensions = []
if not NO_TRAIN:
    torch_sources = [
        "opengigaflow/extensions/custom_gae.cpp",
    ]
    if BUILD_CUDA_EXT:
        extension = CUDAExtension
        torch_sources.append("opengigaflow/extensions/cuda/custom_gae.cu")
    else:
        extension = CppExtension

    torch_extensions = [
        extension(
            "opengigaflow._C",
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

for key, value in cfg_vars.items():
    if value and "-fno-strict-overflow" in str(value):
        cfg_vars[key] = value.replace("-fno-strict-overflow", "")

install_requires = [
    "setuptools<81",
    "numpy",
    "pyyaml",
]

if not NO_TRAIN:
    install_requires += [
        "torch",
        "psutil",
        "rich",
        "hydra-core",
        "omegaconf",
        "pandas",
        "pyarrow",
        "tqdm",
        "matplotlib==3.11.0",
        "wandb",
        "wandb-workspaces",
        "tensorboard",
    ]

setup(
    version="3.0.0",
    # Scope discovery to opengigaflow, does not follow symlinks
    packages=["opengigaflow"]
    + ["opengigaflow." + pkg for pkg in find_namespace_packages(where="opengigaflow")]
    + c_extension_paths
    + ["opengigaflow/extensions"],
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=c_extensions + torch_extensions,
    cmdclass={
        "build_ext": BuildExt,
        "build_torch": TorchBuildExt.with_options(use_ninja=cpp_extension.is_ninja_available()),
        "build_c": CBuildExt,
    },
    include_dirs=[numpy.get_include()],
)
