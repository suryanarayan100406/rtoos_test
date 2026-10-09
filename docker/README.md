# DRISHTI Docker images

Two images, one package. They differ **only** in the compute runtime — the code, CLI, and
bundle format are identical, so a run can start on one tier and finish on the other.

| Image           | Base                              | For                                             | GPU |
|-----------------|-----------------------------------|-------------------------------------------------|-----|
| `drishti:local` | `python:3.11-slim`                | The ground station — Intel Core Ultra 7 155H, no dGPU. CPU + OpenVINO. | none |
| `drishti:cloud` | `nvidia/cuda:12.1.1-cudnn8-runtime`| The T4 tier — Colab / Kaggle / any ~16 GB NVIDIA. Heavy neural stages. | CUDA |

Both are built from the **repository root** (the build context needs `pyproject.toml`, `src/`,
`configs/`, `server/`):

```bash
docker build -f docker/local.Dockerfile -t drishti:local .
docker build -f docker/cloud.Dockerfile -t drishti:cloud .   # needs the NVIDIA Container Toolkit to RUN
```

## Sanity check first — always run the doctor

The doctor probes real hardware and reports which stages are ready. It never fabricates: if a
dependency or GPU is absent it says so.

```bash
docker run --rm drishti:local doctor
docker run --rm --gpus all drishti:cloud doctor      # expect cuda: true, cuda_name: "Tesla T4"
```

## Processing a mission

Mount your inputs read-only and a writable runs directory; point `DRISHTI_RUNS_DIR` at the latter.

```bash
# Full run on the ground (CPU-only stages + anything OpenVINO can handle):
docker run --rm \
  -v "$PWD/data:/data:ro" -v "$PWD/runs:/runs" -e DRISHTI_RUNS_DIR=/runs \
  drishti:local run /data/mission.yaml --profile balanced

# Split across tiers: run the GPU-heavy stages on the T4 box, then finish locally.
# The bundle is resumable — stages already 'done' are skipped by content hash.
docker run --rm --gpus all \
  -v "$PWD/data:/data:ro" -v "$PWD/runs:/runs" -e DRISHTI_RUNS_DIR=/runs \
  drishti:cloud run /data/mission.yaml --profile max --upto s7_dense

docker run --rm \
  -v "$PWD/data:/data:ro" -v "$PWD/runs:/runs" -e DRISHTI_RUNS_DIR=/runs \
  drishti:local run /data/mission.yaml --profile max          # resumes at s8_mesh
```

> On Colab/Kaggle you normally won't build `drishti:cloud` — use the notebook in `notebooks/`,
> which installs the same package into the hosted T4 session. The cloud image is the reproducible,
> **offline-capable** equivalent for a self-hosted GPU.

## Serving the API + viewer

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/runs:/runs" -e DRISHTI_RUNS_DIR=/runs -e DRISHTI_HOST=0.0.0.0 \
  --entrypoint python drishti:local -m server.app
# -> http://localhost:8000/api/health , /api/doctor , /api/runs
```

If `viewer/dist` was built and copied into the image (or mounted at `/app/viewer/dist`), the API
serves the single-page app at `/`. The images are intentionally Python-only and do **not** bundle a
Node toolchain; build the viewer separately (`cd viewer && npm ci && npm run build`) and mount its
`dist/`, e.g. `-v "$PWD/viewer/dist:/app/viewer/dist:ro"`.

## What is NOT baked in, and why

- **GLOMAP binary** — the global mapper is not on PyPI. `pycolmap` (SfM) is installed via the
  `poses` extra; add a GLOMAP build to the image only if you want that specific global-mapping path.
- **Model weights** — downloaded on first use into `HF_HOME` / `OPENVINO_CACHE_DIR`
  (`/cache/...`). Mount a volume there to persist them across runs and to stay offline after a
  one-time warm-up:  `-v "$PWD/cache:/cache"`.
- **Your data and run outputs** — never copied into an image; always mounted.
