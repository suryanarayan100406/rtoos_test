# DRISHTI — CLOUD image (CUDA), targets the free-tier T4 GPU (Colab / Kaggle / any ~16 GB NVIDIA).
#
# This image runs the GPU-heavy neural stages that the local (CPU) tier offloads: learned features /
# poses, promptable masking, and dense depth. It is the SAME package and CLI as the local image — only
# the runtime (CUDA torch + GPU) differs — so a bundle produced here is byte-compatible with the local
# tier and can be finished/served on the ground.
#
# Requires the NVIDIA Container Toolkit on the host (`--gpus all`). On Colab/Kaggle you typically won't
# build this image; use notebooks/ instead. This Dockerfile is the reproducible, offline-capable
# equivalent of that environment for a self-hosted T4/A10/L4 box.
#
# Build from the REPO ROOT:
#   docker build -f docker/cloud.Dockerfile -t drishti:cloud .
#
# Verify the GPU is visible to the pipeline:
#   docker run --rm --gpus all drishti:cloud doctor        # -> cuda: true, cuda_name: "Tesla T4"
#
# Run the GPU stages of a mission, then hand the bundle back to the local tier:
#   docker run --rm --gpus all \
#     -v "$PWD/data:/data:ro" -v "$PWD/runs:/runs" -e DRISHTI_RUNS_DIR=/runs \
#     drishti:cloud run /data/mission.yaml --profile max --upto s7_dense

# CUDA 12.1 runtime + cuDNN, matching the default PyTorch cu121 wheels. "runtime" (not "devel") keeps
# the image lean; we build no CUDA kernels from source here.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- OS + Python 3.11 --------------------------------------------------------------------------
# Ubuntu 22.04 ships Python 3.10; add the deadsnakes PPA for 3.11 to match requires-python and the
# local image exactly (one interpreter version across both tiers = reproducible bundles).
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        ffmpeg libgl1 libglib2.0-0 \
        git curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DRISHTI_RUNS_DIR=/runs \
    HF_HOME=/cache/hf \
    # T4 is compute capability 7.5; pin so libraries that JIT-compile don't probe for absent arches.
    TORCH_CUDA_ARCH_LIST=7.5

WORKDIR /app

# --- Python deps -------------------------------------------------------------------------------
COPY pyproject.toml README.md ./
COPY src/ ./src/

# CUDA build of PyTorch FIRST (default PyPI wheels are cu121), so `.[depth]` reuses it instead of a
# CPU rebuild. Then the full heavy stack: features/poses, masking, dense depth, geo, recon, server.
RUN python -m pip install --upgrade pip \
    && pip install "torch>=2.2" \
    && pip install ".[video,geo,recon,poses,spine,telem,depth,server]"

COPY configs/ ./configs/
COPY server/ ./server/

RUN mkdir -p /runs /cache/hf

# Bare `docker run --gpus all drishti:cloud` prints the doctor — confirms CUDA is actually visible
# before you spend GPU minutes on a run.
ENTRYPOINT ["drishti"]
CMD ["doctor"]
