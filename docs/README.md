# DRISHTI

**D**rone-based **R**eal-time **I**maging for **S**ingle-pass **H**igh-fidelity **T**errain **I**ntelligence

> *One pass. The whole picture — a georeferenced, measurable 3D model of everything the drone flew
> over, reconstructed offline from the recorded mission.*

**Problem:** NTRO / Smart India Hackathon — **PS-17 / SIH26158** · Category: Software · Theme: Drone /
Robotics — *"Single-Pass Drone Video to Accurate 3D Model Generation System."* Generate a
georeferenced, metrically accurate, textured 3D model from a **single** drone pass — no multi-pass
grids, no extensive Ground Control Points, with offline post-flight processing in this build.

> **Scope note:** The evaluator-facing theory, pitch, and canonical architecture documents retain the
> full long-term vision, including the optional Edge Tier/live path. The implemented build is the recorded-
> footage, offline, ground/cloud realization. For exact current capabilities and operator instructions, use
> [`BUILD-STATUS-AND-USAGE.md`](BUILD-STATUS-AND-USAGE.md) and [`GUIDE.md`](GUIDE.md).

---

## The idea in one paragraph

Classical drone mapping needs many overlapping passes and hours of Structure-from-Motion + Multi-View
Stereo — it cannot handle the "one chance to fly" reality of disaster response, reconnaissance and
rapid mapping. DRISHTI reframes the problem: a single pass gives **weak multi-view geometry** but a
**strong GNSS + temporal signal (inertial when an IMU is present) plus learned priors**. The current
shipped realization processes the recorded mission offline on the Ground/Cloud tiers: it combines
classical visual geometry, metric-depth priors, and GNSS alignment into one resumable full-quality path.
Every stage emits confidence/evidence metadata and a fallback or explicit failure reason. The live Edge
Tier and streaming path remain canonical architecture references, not part of this build.

---

## How to read this documentation

- **Judges / NTRO evaluators (15 min):** start with **[Master Overview](00-MASTER-OVERVIEW.md)** →
  **[Pitch & Presentation](06-PITCH-AND-PRESENTATION.md)**.
- **Technical reviewers:** **[Theory](01-THEORY.md)** → **[How It Works](02-HOW-IT-WORKS.md)** →
  **[Technology Stack](03-TECHNOLOGY-STACK.md)** → **[Integration](04-INTEGRATION.md)** →
  **[Design Decisions](05-DESIGN-DECISIONS.md)**.
- **Build team:** your **[role document](#role-documents)** + How It Works + Integration.

---

## Document map

### Core documents

| Doc | What it covers |
|-----|----------------|
| [00 · Master Overview](00-MASTER-OVERVIEW.md) | The whole system at a glance; thesis, architecture, outputs, evaluation, applications, roadmap, risks. Start here. |
| [01 · Theory & Scientific Foundations](01-THEORY.md) | The math/science: camera geometry, why single-pass is hard, SfM/BA, monocular & feed-forward depth, VIO fusion, GNSS/RTK, georeferencing, 3DGS, TSDF, accuracy theory. |
| [02 · How It Works](02-HOW-IT-WORKS.md) | End-to-end pipeline, stage by stage (S0–S10), the two output paths, a worked example, data-flow, challenge→mechanism traceability. |
| [03 · Technology Stack](03-TECHNOLOGY-STACK.md) | Concrete tools/models/hardware with versions, licenses, rationale, alternatives; BOM; sovereignty/air-gap notes. |
| [04 · Integration](04-INTEGRATION.md) | How components work together: interfaces, message contracts, coordinate frames, time sync, edge↔ground protocol, hardware/software plumbing, failure handling. |
| [05 · Design Decisions](05-DESIGN-DECISIONS.md) | Every major decision with options considered, rationale, trade-offs, and risks. |
| [06 · Pitch & Presentation](06-PITCH-AND-PRESENTATION.md) | Elevator + 3-min scripts, slide-by-slide deck, demo storyboard, judge Q&A. |

### Role documents

| Role | Owns (pipeline stages) |
|------|------------------------|
| [1 · 3D Reconstruction Research Lead](roles/1-3d-reconstruction-lead.md) | Reconstruction backbone: S4 geometry, S5 fusion, S6 global opt, S7 dense/3DGS, S8 mesh/texture |
| [2 · Computer Vision & Video Intelligence](roles/2-computer-vision-video-intelligence.md) | Front-end: S1 ingest/QA/keyframes, matching, flow, S3 dynamic + semantic masking |
| [3 · AI / Deep Learning Research](roles/3-ai-deep-learning-research.md) | Learned models: metric depth, pointmaps, segmentation, completion, confidence, on-device optimization |
| [4 · Drone & Sensor / Hardware Integration](roles/4-drone-sensor-hardware-integration.md) | Capture system: S0 capture/sync, platform, sensors, calibration, comms, edge compute integration |
| [5 · Geospatial & Accuracy Research](roles/5-geospatial-accuracy-research.md) | Metric spine, GNSS/RTK factors (S6), S9 georeferencing/DSM/ortho, accuracy validation, formats |
| [6 · Systems & Edge / Compute Optimization](roles/6-systems-edge-compute-optimization.md) | Runtime: tiering, ROS 2, TensorRT, streaming, reliability spine, deployment |

### Implementation (build docs)

> The **build** we are actually shipping: recorded footage + telemetry → one full-quality georeferenced
> 3D model, **offline and ground-only** (no drone/edge deployment, no live path), on modest + free
> compute. This is a ground-only realization of the canonical architecture — see the scope delta in
> [`../AGENTS.md` §2](../AGENTS.md).

| Doc | What it covers |
|-----|----------------|
| [`../notebooks/KAGGLE-GUIDE.md`](../notebooks/KAGGLE-GUIDE.md) | **First-time Kaggle walkthrough:** setup, every notebook section, sample/personal inputs, resume, downloads, troubleshooting, and local viewing. |
| [`GUIDE.md`](GUIDE.md) | **Complete usage + how-it-works guide for the shipped build (start here to *use* it):** install, prepare inputs, run/resume, local↔T4 tiering, outputs, serve/view, and every pipeline stage in detail — code-accurate. |
| [`BUILD-STATUS-AND-USAGE.md`](BUILD-STATUS-AND-USAGE.md) | Audited build status, exact operator workflow, validation results, and remaining native/cloud proof. |
| [`../AGENTS.md`](../AGENTS.md) | **Living context for AI agents (read first):** scope, constraints, principles, status, decision log |
| [implementation/00 · PRD](implementation/00-PRD.md) | Product requirements: requirements, outputs, success metrics, acceptance criteria |
| [implementation/01 · Implementation Plan](implementation/01-IMPLEMENTATION-PLAN.md) | All seven phases (0–6): goals, tasks, local/cloud compute, exit criteria |
| [implementation/02 · System Design](implementation/02-SYSTEM-DESIGN.md) | Code architecture: project-bundle model, stage interface, config, compute tiering, model registry |
| [implementation/README](implementation/README.md) | Index + reading paths for the build docs |

### Reference (internal)

| File | Purpose |
|------|---------|
| [Problem statement + Output/Evaluation tables](_internal/PROBLEM_STATEMENT.md) | Canonical problem + the two required tables |
| [Canonical architecture spec](_internal/CANONICAL-ARCHITECTURE-SPEC.md) | Single source of truth for tiers, stages, models, spines |
| [Style guide](_internal/STYLE_GUIDE.md) | Authoring conventions & honesty policy |
| `_internal/research/*.md` | Six domain research dossiers underpinning the design |

---

## Quick facts

- **Canonical architecture:** three tiers and two output paths are documented for the full vision. **Shipped
  build:** offline recorded-footage processing on local CPU/OpenVINO and optional Cloud T4, with one
  full-quality path; there is no on-UAV compute or live path.
- **Inputs (official contract):** *mandatory* — drone video (1080p/4K), GPS coordinates, flight metadata.
  *Optional* — IMU, barometric altitude, camera intrinsics, RTK/PPK. Everything here is designed to run on
  the **mandatory-only** capture, with intrinsics self-calibrated; the optional streams tighten the solve
  where they exist and are never assumed.
- **Accuracy (design targets):** official bar **≤ 1 m** absolute — RTK/PPK → 3–8 cm horizontal (meets
  it comfortably); GPS-only baseline → target sub-metre-to-≤ 1 m, regions > 1 m flagged; GSD ≈ 2 cm/px.
- **Speed (design target):** **< 15 min for a 10-minute video** on suitable GPU-class ground hardware;
  current hardware-specific timing remains to be measured honestly.
- **Reliability:** graceful-degradation ladder L0–L6 where a real fallback exists; absent required native
  tools fail loudly, and optional exports are recorded as skipped with reasons.
- **Outputs:** LAS/LAZ/PLY point cloud, OBJ/glTF-GLB/**FBX** mesh, optional textured GLB/3DGS artifacts,
  DSM/DTM + orthomosaic (GeoTIFF), evidence/accuracy report, and an offline web viewer. Semantic
  propagation and region-aware measurement authority remain incomplete acceptance work.
- **Graded on (official weights):** reconstruction accuracy **30%** · model completeness **20%** ·
  processing speed **20%** · innovation **15%** · scalability **10%** · user interface **5%** — quoted
  verbatim in the [problem statement](_internal/PROBLEM_STATEMENT.md) §1a, which is also where every
  target above is sourced from.


---

*See the [Master Overview](00-MASTER-OVERVIEW.md) for the full story.*
