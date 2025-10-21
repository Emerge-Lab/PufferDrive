FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG PYVER="3.12"
# Set environment variables for the build
ENV TORCH_CUDA_ARCH_LIST=Turing \
    TZ=Europe/Paris\
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install all system dependencies required for the build process
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
    software-properties-common \
    # git \
    curl \
    # autoconf \
    # libtool \
    # flex \
    # bison \
    # libbz2-dev \
    build-essential \
    # htop \
    #clang \
    # gdb \
    # llvm \
    # tmux \
    # psmisc \
    # sudo \
    # libglfw3-dev \
    # ninja-build \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python$PYVER python$PYVER-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


WORKDIR /pufferdrive
#Change WORKDIR place

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Configure Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python$PYVER 1 \
    && update-alternatives --set python /usr/bin/python$PYVER \
    && python -m pip install --no-cache-dir --upgrade pip


RUN pip install --no-cache-dir setuptools && \
    pip install --no-cache-dir wheel && \
    pip install --no-cache-dir cython && \
    pip install --no-cache-dir tensorflow-cpu && \
    pip install --no-cache-dir tensorflow.io && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    pip install --no-cache-dir gcsfs

COPY pyproject.toml setup.py setup.cfg MANIFEST.in ./
#Todo check if needed
COPY pufferlib ./pufferlib

RUN pip install --no-cache-dir -e .
