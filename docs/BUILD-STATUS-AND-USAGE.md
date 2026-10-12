# DRISHTI build status and usage

**Purpose:** Give a code-accurate answer to “what is built, how do I run it, and what still needs a
native/cloud proof?”

**Audience:** DRISHTI users, evaluators, operators, and contributors who need one reliable starting point.

## TL;DR

- DRISHTI is implemented as an offline, ground-only pipeline from recorded video + telemetry to a
  resumable project bundle.
- The CLI, configuration system, bundle/manifest, all S0–S10 stages, report generation, server, and
  offline Cesium viewer are present.
- The pure/core implementation is locally verified: **146 tests**, Ruff clean, Python compilation clean,
  valid notebooks, and a passing viewer typecheck/build.
- A full real-footage run is **not proven on this workstation** because CUDA, Open3D, pycolmap, PDAL/GDAL,
  and the learned checkpoints are not installed here. Use the T4 notebooks for that proof.
- No accuracy or processing-time target is considered measured until a real run and, ideally, held-out
  survey check-points are processed.

## 1. What “built” means

### Built and locally verified

| Area | Current implementation |
|---|---|
| CLI | `run`, `resume`, `inspect`, `verify`, `doctor`, and `stages` |
| Configuration | YAML defaults/profiles, dataset descriptors, `DRISHTI__...` environment overrides, `--set` overrides, manifest snapshots |
| Bundle | Versioned on-disk bundle, stage freshness, atomic manifest writes, output hashing, resume support |
| S0–S1 | Video/telemetry ingest, CRS derivation, frame QA, keyframe selection |
| S2 | COLMAP/pycolmap SfM, GPS Sim(3) alignment, optional GTSAM visual/GNSS graph |
| S3 | Real RT-DETR box detection and dynamic masks, explicit capability/degradation metadata |
| S4 | Depth Anything V2, optional Apache DA3/MapAnything adapters, metric anchoring, confidence/disagreement artifacts |
| S6 | Lens rectification, global views, optional real GLOMAP execution and CRS alignment |
| S7 | Spatial/adaptive TSDF tiling, streaming PLY output, optional CUDA/gsplat route, tile evidence |
| S8 | Poisson meshing, low-density cropping, best-view color, optional xatlas depth-visible textured GLB |
| S9 | Streaming DSM/DTM/orthomosaic rasterization, optional executed PDAL SMRF, GeoTIFF output |
| S10 | OBJ/PLY/LAS/LAZ/glTF/GLB/FBX/3D Tiles/Potree attempts with post-write validation and reasons |
| Report | HTML/JSON report with timings, confidence, degradation, CRS, geometry evidence, and validation status |
| Viewer/server | FastAPI manifest/artifact API and offline React/Vite/Cesium viewer |

### Not yet proven as a complete product run

- S2–S10 over a fresh real `aukerman` or `waterbury` bundle on the current code revision.
- DA3, MapAnything, gsplat, **Open3D 0.19.0**, pycolmap, GLOMAP, PDAL, Blender, and all export tools together in one
  native environment.
- ≤1 m accuracy. This is an acceptance target, not a current measured result.
- The official `<15 min for a 10-minute video` target. Runtime must be reported for the actual hardware.
- SAM2 pixel refinement, ByteTrack tracking, RAFT motion residuals, and semantic propagation. S3 records
  these as requested-but-not-executed rather than claiming them.
- IMU preintegration, RTK covariance/state factors, lever-arm correction, and time-offset estimation.
- Full MVS seam optimization. The available atlas path is xatlas best-view depth-visible baking.

## 2. Repository map

```text
src/drishti/          Python package and CLI
configs/              defaults, profiles, and dataset descriptors
tests/                pure logic, contract, mocked-native-boundary tests
notebooks/            Colab/Kaggle full and tier-split runners
server/               FastAPI app
viewer/               React + Vite + Cesium viewer
docs/GUIDE.md         detailed shipped-code guide
docs/RUN_GUIDE.md     cloud/Kaggle operational walkthrough
scripts/              realistic/synthetic input generation helpers
runs/                 generated bundles; git-ignored
data/                 user/sample inputs; git-ignored except README
```

## 3. Install the light local spine

Use Python 3.11 or 3.12. Open3D currently requires a Python 3.12-compatible wheel in the hosted path.

### Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[video,geo,telem,dev]"
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[video,geo,telem,dev]"
```

For a local heavy installation, add the appropriate groups:

```bash
pip install -e ".[video,geo,recon,texture,poses,spine,telem,depth,export,server,dev]"
```

Use the hosted notebooks instead when CUDA/native wheels are unavailable. The notebooks install core first,
then install heavy groups separately so one unavailable wheel does not hide the CLI. Hosted acquisition
explicitly sets `DRISHTI__models__allow_download=true`; the normal local default is offline (`false`).

## 4. Prepare a mission

Create a descriptor, for example `configs/datasets/my_mission.yaml`:

```yaml
name: my_mission
video: data/my_mission/flight.MP4
telemetry:
  format: dji_srt       # dji_srt | csv | mavlink | exif
  path: data/my_mission/flight.SRT
  # csv:                 # required when format=csv
  #   time: timestamp
  #   lat: latitude
  #   lon: longitude
  #   alt: abs_alt
optional:
  imu: null
  baro: null
  intrinsics: null
  rtk: null
crs:
  mode: derive_from_gps
  epsg: null
report:
  check_points: null
```

Required inputs are a recorded video and real telemetry with a usable GNSS track. Optional intrinsics,
IMU, barometer, RTK/PPK, and surveyed check-points improve the solve or validate it; they are not silently
invented when absent.

The descriptor paths are resolved from the process working directory. Absolute paths are safest for files
outside the repository. The CRS is derived from the median longitude/latitude unless an explicit EPSG is
provided.

## 5. Run and inspect

Start with readiness checks:

```bash
drishti doctor
drishti stages
```

Run a complete mission:

```bash
drishti run --dataset configs/datasets/my_mission.yaml --profile balanced
```

Useful controlled runs:

```bash
# Run only through a stage from the beginning.
drishti run --dataset configs/datasets/my_mission.yaml --profile balanced --upto s2_poses

# Run one stage after its dependencies already exist in the bundle.
drishti run --dataset configs/datasets/my_mission.yaml --only s4_depth

# Override values without editing YAML.
drishti run --dataset configs/datasets/my_mission.yaml --profile balanced \
  --set poses.matcher=exhaustive \
  --set dense.depth_trunc_m=150

# Continue a stopped run. The recorded bundle config is reused.
drishti resume runs/<run_id>

# Inspect and verify.
drishti inspect runs/<run_id>
drishti verify runs/<run_id>
```

Configuration precedence is:

```text
--set override > DRISHTI__section__key environment variable > dataset CRS/report
> selected profile > configs/default.yaml
```

Examples:

```bash
export DRISHTI__models__allow_download=true   # hosted acquisition only
export DRISHTI__ingest__target_fps=2
```

The bundle is written to `runs/<run_id>/`. Each stage records status, environment, timing, output paths,
output hash, measured metrics, confidence, and degradation reason in `manifest.json`.

## 6. Recommended cloud workflows

### Full hosted run

**First-time Kaggle user:** follow the complete
[`notebooks/KAGGLE-GUIDE.md`](../notebooks/KAGGLE-GUIDE.md) walkthrough, including writable source setup,
personal inputs, exact resume limitations, export checks, archive handling, and local viewing.

Open `notebooks/drishti_colab_full.ipynb` on Colab or Kaggle with a GPU accelerator. It:

1. Installs the package and heavy dependencies, including a Python 3.12 environment when required.
2. Acquires pinned checkpoints with download permission explicitly enabled.
3. Imports a generated sample, Drive mission, uploaded mission, or partial bundle.
4. Runs all stages and tees logs to the bundle.
5. Runs `inspect`/`verify` and lets you download the complete or slim bundle.

For high-altitude aerial data, use its high-RAM 6B path with exhaustive matching and a depth truncation
that exceeds the measured AGL/slant range. Do not use the 40 m low-altitude path for a 100 m AGL survey.

### Tier-split run

Use `notebooks/drishti_cloud_t4.ipynb` when the local machine has the light dependencies and you want to
run GPU-heavy stages on a T4, download the partial bundle, and resume locally from `s8_mesh` onward.

## 7. Bundle outputs

Important artifacts include:

```text
runs/<id>/
├── manifest.json
├── s0_ingest/          frames, telemetry, CRS
├── s1_frameqa/         QA metrics and keyframes
├── s2_poses/           poses, sparse points, SfM artifacts
├── s3_masking/         masks.json, mask PNGs, capability metadata
├── s4_depth/           depth.json, metric depth/confidence arrays
├── s6_global/          global poses, rectified views, sparse points
├── s7_dense/           dense.ply or gaussians, tile_states.json
├── s8_mesh/            mesh.ply, mesh.json, texture.json, optional textured.glb
├── s9_geo/             dsm.tif, dtm.tif, optional ortho.tif, geo.json
├── s10_export/         requested deliverables and exports.json
├── report/              report.json and report.html
└── logs/                stage/run logs
```

`measurement_authoritative` is deliberately conservative in the current reconstruction artifacts. A
plausible-looking model is not automatically a survey-validated measurement product. Use check-points and
the report before making accuracy claims.

## 8. Serve and view a finished bundle

Install the server extra and start the API from the repository root:

```bash
pip install -e ".[server]"
python -m server.app
```

Build/run the viewer in a second shell:

```bash
cd viewer
npm ci
npm run typecheck
npm run build
npm run dev
```

The API uses the on-disk manifest as its source of truth. The viewer renders the available georeferenced
3D Tiles/model artifacts and exposes run/status information. Potree point-cloud rendering is optional and
is not claimed when its converter/renderer is absent.

## 9. What the local checks prove

The repository’s repeatable local checks are:

```bash
PYTHONPATH=src /tmp/omnirush/rtos-venv/bin/python -m pytest -q
/tmp/omnirush/rtos-venv/bin/ruff check src tests
/tmp/omnirush/rtos-venv/bin/mypy src server
python3 -m py_compile $(python3 - <<'PY'
from pathlib import Path
for path in Path("src").rglob("*.py"):
    print(path)
PY
)
cd viewer && npm run typecheck && npm run build
```

These checks cover pure logic, bundle/config contracts, reliability policy, mocked native boundaries,
export/raster integrity, Python syntax, and frontend compilation. They do not replace a native/cloud
full-pipeline run.

## 10. Audit result and remaining proof

The current audit found no shipped fake-output path. The abstract `Stage.run` `NotImplementedError` is an
interface guard, and missing native tools fail loudly or produce an explicitly degraded/ skipped format.

Remaining high-impact work:

1. Run balanced TSDF on real `aukerman` and `waterbury` with native/cloud dependencies.
2. Run and measure Apache DA3/MapAnything and the max Gaussian path on CUDA.
3. Wire SAM2, ByteTrack, RAFT, semantic propagation, and IMU/RTK calibrated factors when their actual
   environment contracts are available.
4. Validate all exports, atlas behavior, memory peaks, timings, tile evidence, and held-out check-point
   accuracy.
5. Reconcile any native-run failures back into this document and `AGENTS.md`; do not turn design targets
   into measured claims.

## Open questions / risks

- Aerial domain performance of learned depth/geometry remains unmeasured on the acceptance datasets.
- Free-tier sessions can fail from host RAM, GPU memory, or time limits on large missions.
- Building backs, occluded surfaces, and low-parallax regions remain incomplete or low-confidence by nature.
- Export availability depends on optional system tools such as Blender, PDAL, py3dtiles, and PotreeConverter.
