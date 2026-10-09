# DRISHTI

**Single-pass drone video → accurate, georeferenced 3D model.** Offline, ground-only realization of
the DRISHTI architecture: you hand it a *recorded* drone clip + GPS + flight metadata, and it produces
one full-quality georeferenced 3D model (mesh + point cloud + DSM/DTM + orthomosaic + semantic layers),
an honest accuracy report, and a web viewer.

> NTRO / Smart India Hackathon **PS-17 · SIH26158** — *Single-Pass Drone Video to Accurate 3D Model
> Generation System.*

## Read this first

- **[`notebooks/KAGGLE-GUIDE.md`](notebooks/KAGGLE-GUIDE.md)** — **new to DRISHTI? Start here for Kaggle:**
  account/setup, every notebook section, real sample or personal footage, recovery, downloads, and
  local model viewing, with copyable commands and explanations.
- **[`docs/GUIDE.md`](docs/GUIDE.md)** — the complete **usage + how-it-works** guide: install, prepare
  inputs, run/resume, local↔T4 tiering, outputs, serve/view, and every pipeline stage in detail
  (code-accurate). **Want to actually use DRISHTI? Start here.**
- **[`docs/BUILD-STATUS-AND-USAGE.md`](docs/BUILD-STATUS-AND-USAGE.md)** — the audited build status,
  exact usage workflow, validation results, and remaining native/cloud proof.
- **[`AGENTS.md`](AGENTS.md)** — the living project context (scope, constraints, principles, status,
  decision log). **AI agents and new contributors start here.**
- **[`docs/implementation/`](docs/implementation/)** — PRD, full phased implementation plan, system design.
- **[`docs/`](docs/)** — the evaluator-facing story (overview, theory, how-it-works, tech stack).

## What this build is (and isn't)

- ✅ Recorded footage + telemetry **in** → one full-quality georeferenced 3D model **out**.
- ✅ **Offline, ground-only.** Light stages run **local** (Intel Ultra 7 155H, CPU + Arc iGPU/NPU via
  OpenVINO); heavy neural stages tier to **free cloud GPU** (Colab/Kaggle T4).
- ❌ No on-drone / edge deployment, no Jetson, no live/streaming path. (See [`AGENTS.md` §2](AGENTS.md).)
- **No hardcoded data, no stub outputs.** Real footage → real models → real artifacts; missing inputs
  or absent dependencies **fail loudly** rather than fabricate.

## Install

```bash
# core (light) spine — runs the CLI, config, bundle, telemetry parsing, planning
python -m venv .venv && . .venv/Scripts/activate     # Windows: .venv\Scripts\activate
pip install -e .

# add capability groups per environment (see pyproject.toml [optional-dependencies])
pip install -e ".[video,geo,recon,poses,spine,telem,depth,server,dev]"
```

Heavy groups (`recon`, `poses`, `spine`, `depth`) need native libraries / a GPU and are installed on
the machine that runs those stages (local workstation or a Colab/Kaggle session). The core spine
installs everywhere.

> `recon` pulls **Open3D**, which has wheels only through **Python 3.12**. On Python 3.13+ (today's
> Colab/Kaggle) install fails — use the notebooks in [`notebooks/`](notebooks/), which auto-build a 3.12
> environment. See [`docs/GUIDE.md`](docs/GUIDE.md) §2.

## Prepare your data (where your footage + metadata go)

DRISHTI reads a **dataset descriptor** — a small YAML file that points at your real files. It ships
**no** sample data; missing mandatory inputs fail loudly (never a placeholder).

1. Put your files anywhere. The `data/` folder is a convenient, git-ignored drop zone:

   ```
   data/my_mission/
   ├── flight.MP4     # your single-pass drone clip (1080p/4K)
   └── flight.SRT     # the matching telemetry (see formats below)
   ```

2. Copy the template descriptor and edit the paths:

   ```bash
   cp configs/datasets/example.yaml configs/datasets/my_mission.yaml
   ```

   ```yaml
   name: my_mission
   video: data/my_mission/flight.MP4        # path is relative to where you run `drishti`
   telemetry:
     format: dji_srt                         # dji_srt | csv | mavlink | exif
     path: data/my_mission/flight.SRT
   optional:                                 # all optional; used only when present
     imu: null
     baro: null
     intrinsics: null                        # camera-intrinsics yaml/json, else self-calibrated
     rtk: null                               # RTK/PPK-corrected track for higher accuracy
   crs:
     mode: derive_from_gps                   # default; use mode: epsg + epsg: <code> to force one
     epsg: null
   report:
     check_points: null                      # surveyed points (GeoJSON/CSV) => validated accuracy
   ```

**Telemetry formats** (your flight metadata):

| `format`  | You provide                                          | Needs |
|-----------|------------------------------------------------------|-------|
| `dji_srt` | DJI `.SRT` sidecar (GPS/alt per frame)               | core (built-in) |
| `csv`     | a `.csv` track + a `csv:` column mapping             | core (built-in) |
| `mavlink` | ArduPilot/PX4 `.tlog` or `.bin` log                  | `telem` extra (pymavlink) |
| `exif`    | a GPS-tagged video, or a folder of geotagged frames  | `exiftool` on PATH |

For `csv`, add the column mapping under `telemetry`:

```yaml
telemetry:
  format: csv
  path: data/my_mission/track.csv
  csv:
    time: timestamp     # seconds, or ISO-8601 (auto-detected)
    lat: latitude
    lon: longitude
    alt: abs_alt
    yaw: yaw            # optional (pitch/roll likewise)
```

## Run it

```bash
drishti doctor                                     # what compute + which stages are ready HERE
drishti stages                                     # the canonical pipeline order
drishti run --dataset configs/datasets/my_mission.yaml --profile balanced
```

Other commands — note the bundle path is a **positional argument**, not a flag:

```bash
drishti run     --dataset <d.yaml> --profile fast  # profile: fast | balanced | max
drishti run     --dataset <d.yaml> --upto s2_poses # run from the start THROUGH one stage
drishti run     --dataset <d.yaml> --only s4_depth # run a single stage (its deps must be done)
drishti resume  runs/<id>                          # continue an interrupted/partial bundle
drishti inspect runs/<id>                          # print the manifest (status, timings, confidence)
drishti verify  runs/<id>                          # recompute output hashes vs. the manifest
```

A run produces a **project bundle** at `runs/<id>/` — a versioned, content-hashed directory holding
every stage's artifacts, a `manifest.json` with provenance and **measured** timings/confidence, and
the accuracy report. It is **resumable**: re-running skips stages whose inputs/params are unchanged.

## What runs where (no local GPU)

The dev machine (Intel Ultra 7 155H, no discrete GPU) runs the light stages; the heavy neural stages
belong on a **free cloud T4**. Bundles are byte-compatible across tiers, so a run can split.

 | Stage(s) | Where | Extra to install |
|----------|-------|------------------|
| `s0_ingest` (decode, telemetry, CRS), `s1_frameqa` | local CPU | `video`, `telem` |
 | `s6_global` | local CPU | `video`, `geo` |
 | `report` | local CPU | core only (plus `geo` when validating check-points) |
| `s2_poses`, `s7_dense`, `s8_mesh` | local (heavy) or T4 | `poses`, `spine`, `recon` |
| `s3_masking`, `s4_depth` (neural) | **T4 recommended** | `depth` (+ GPU) |
 | `s9_geo`, `s10_export` | local CPU | `geo`, `recon`, `export` |

- **Fastest real milestone locally:** `pip install -e ".[video,telem]"`, then `drishti run …` — you'll
  watch S0 decode frames, parse your GPS track, and derive the CRS into a real bundle.
- **Full quality, no local GPU — two cloud paths** (both in [`notebooks/`](notebooks/), same package + CLI):
   - **Run everything in the hosted session** — [`notebooks/drishti_colab_full.ipynb`](notebooks/drishti_colab_full.ipynb)
     runs all 11 stages using the host CPU/RAM and GPU as appropriate and hands back the bundle
     (model + deliverables + report + logs); locally you only *view/serve* it. Follow the
     [complete Kaggle guide](notebooks/KAGGLE-GUIDE.md).
  - **Tier-split** — [`notebooks/drishti_cloud_t4.ipynb`](notebooks/drishti_cloud_t4.ipynb) (or
    `docker/cloud.Dockerfile`) runs the heavy stages on the T4, then you `drishti resume` the light
    stages locally.

## API server + web viewer (optional)

```bash
pip install -e ".[server]"
python -m server.app                               # http://localhost:8000/api/health

cd viewer && npm install && npm run dev            # http://localhost:5173 (proxies /api → :8000)
```

The viewer lists runs, shows the live per-stage table + doctor, and renders the georeferenced 3D
Tiles model in the browser (offline CesiumJS). See [`viewer/README.md`](viewer/README.md).

## Repo layout

See [`AGENTS.md` §7](AGENTS.md) and [`docs/implementation/02-SYSTEM-DESIGN.md`](docs/implementation/02-SYSTEM-DESIGN.md).

## License

Code: Apache-2.0. Shipped models are permissively licensed only; non-commercial / military-excluded
models are reference-only (see [`docs/03-TECHNOLOGY-STACK.md`](docs/03-TECHNOLOGY-STACK.md)).
