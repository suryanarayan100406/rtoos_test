# DRISHTI — LOCAL image (CPU + OpenVINO), targets the Intel Core Ultra 7 155H tier.
#
# This is the "ground station" image: no discrete GPU. Neural stages that CAN run on CPU/iGPU/NPU
# do so through OpenVINO; the heaviest neural stages (dense depth, learned features) are meant to be
# offloaded to the cloud-T4 image instead (see cloud.Dockerfile). Everything else — ingest, frame QA,
# poses (COLMAP/GLOMAP), global optimization, meshing, geo/export, report — runs here, offline.
#
# Build from the REPO ROOT (context must include pyproject.toml + src/):
#   docker build -f docker/local.Dockerfile -t drishti:local .
#
# Run the doctor (prints detected compute + which stages are ready):
#   docker run --rm drishti:local doctor
#
# Process a dataset (mount your data and an output dir):
#   docker run --rm \
#     -v "$PWD/data:/data:ro" -v "$PWD/runs:/runs" \
#     -e DRISHTI_RUNS_DIR=/runs \
#     drishti:local run /data/mission.yaml --profile balanced
#
# Serve the API + built viewer (bind 0.0.0.0 so the port is reachable outside the container):
#   docker run --rm -p 8000:8000 -v "$PWD/runs:/runs" \
#     -e DRISHTI_RUNS_DIR=/runs -e DRISHTI_HOST=0.0.0.0 \
#     --entrypoint python drishti:local -m server.app

FROM python:3.11-slim-bookworm

# --- OS packages -------------------------------------------------------------------------------
# ffmpeg           : PyAV / OpenCV video decode
# libgl1,libglib2  : runtime .so's Open3D and OpenCV-headless still dlopen
# git, ca-certs    : pip VCS installs + TLS
# build-essential  : source builds for any wheel-less transitive dep (kept minimal)
# NOTE: rasterio/pyproj wheels bundle their own GDAL/PROJ, so we do NOT apt-install libgdal here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        git \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keep model/cache downloads inside a mountable, writable location.
    DRISHTI_RUNS_DIR=/runs \
    HF_HOME=/cache/hf \
    OPENVINO_CACHE_DIR=/cache/openvino

WORKDIR /app

# --- Python deps (layer-cached on dependency metadata) ----------------------------------------
# Copy only what defines dependencies first, so edits to source don't bust the heavy pip layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# 1) CPU-only PyTorch FIRST, from the CPU wheel index. Without this, `.[depth]` would pull the
#    default CUDA build of torch (~2.5 GB) which is useless on this GPU-less tier.
RUN python -m pip install --upgrade pip \
    && pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

# 2) The package + the extras that make sense locally. Torch is already satisfied above, so `depth`
#    only adds openvino + onnx here. `poses` installs pycolmap (SfM); the GLOMAP global-mapper
#    *binary* is optional and not on PyPI — install it separately if you want the GLOMAP path.
RUN pip install ".[video,geo,recon,poses,spine,telem,depth,server]"

# --- App + configs -----------------------------------------------------------------------------
COPY configs/ ./configs/
COPY server/ ./server/

# A pre-built viewer (viewer/dist) is served at / by the API when present. It is intentionally NOT
# built inside this image (keeps it Python-only + small); build it separately and mount/copy it, or
# use the multi-stage cloud image note in docker/README.md.

RUN mkdir -p /runs /cache/hf /cache/openvino

# `drishti` is the CLI entrypoint (see [project.scripts]); default to the doctor so a bare
# `docker run drishti:local` is informative and side-effect-free.
ENTRYPOINT ["drishti"]
CMD ["doctor"]
