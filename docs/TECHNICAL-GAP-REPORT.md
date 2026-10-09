# DRISHTI technical gap report

**Audit date:** 2026-10-08  
**Scope:** offline ground/cloud product in this repository  
**Evidence basis:** source inspection, configuration inspection, upstream model audits, and the existing
real-footage run records in `AGENTS.md`.

## Executive findings

The repository already has a real, resumable S0–S10 spine and memory-bounded TSDF/geospatial path. The
largest product risks were semantic rather than missing dependencies: optional tools were sometimes
reported as if they had run, spatial depth clipping could turn sparse-coverage gaps into blank tiles, and
the configured Gaussian profile was an intentional error path.

This change set addresses those correctness risks:

| Area | Before | Now |
|---|---|---|
| GTSAM | availability check only | real GNSS + visual position factor graph when GTSAM is installed; explicit IMU-adapter limitation |
| GLOMAP | executable presence raised confidence | mapper is invoked, output is parsed, aligned back to the S2 CRS, and only then accepted |
| PDAL | import presence labeled DTM `pdal_smrf` | SMRF is executed and classified-point count is recorded; otherwise the report names the fallback |
| Dense tile bounds | one global sparse Z envelope | local sparse envelopes; no local support opens the band rather than deleting valid learned depth |
| Tile gaps | silently skipped | `tile_states.json` records reconstructed vs unknown/unobservable and a reason |
| Learned geometry | only DA V2 wrapper | swappable backend contract plus optional Apache MapAnything and Apache DA3 adapters; DA3 confidence maps are persisted |
| Gaussian profile | `dense.method=gaussian` raised | real CUDA/gsplat optimization path with metric-depth initialization, RGB/depth losses, checkpoints, and a metric Gaussian PLY |

## Remaining high-impact gaps

These are deliberately visible and should not be treated as completed by the presence of an interface:

1. **Aerial benchmark evidence for DA3/MapAnything.** Upstream model cards establish capability and
   licensing, not DRISHTI's aerial accuracy. Add measured nadir/oblique comparisons against checkpoints
   and held-out survey points before making either model the balanced default.
2. **IMU/RTK factor adapters.** The current GTSAM implementation optimizes visual relative motion and
   GNSS position factors. IMU and RTK files are ingested as inputs, but calibrated preintegration,
   covariance/state handling, lever-arm correction, and time-offset estimation remain to be wired.
3. **Dynamic perception.** S3 currently emits real RT-DETR box masks. Its versioned capability metadata
   explicitly records SAM2, tracking, optical-flow residuals, and semantic propagation as not executed;
   those paths remain absent until their actual inference paths are integrated.
4. **Texture atlases.** S8 now produces a real optional xatlas best-view, depth-visible UV atlas and
   textured GLB when its optional dependencies are installed. Full multi-view seam optimization and
   photometric blending remain necessary for a commercial-quality mesh deliverable.
5. **Semantic geometry.** Semantic class configuration exists, but point/mesh/raster label propagation
   is not yet a complete artifact contract.
6. **Viewer analytics and measurement authority.** The current Cesium viewer can display the main model
   artifacts; region-aware confidence overlays and measurement provenance need to consume the new tile and
   geometry-class artifacts.
7. **Real acceptance runs.** Native dependencies and CUDA are not available in this execution environment.
   The Kaggle canonical notebook remains the required validation path for S4–S10 timing, memory, Gaussian
   training, and visual quality.

## Acceptance gates for the next engineering pass

- Run `aukerman` and `waterbury` through the balanced TSDF path and record per-tile states, S2/S6 residuals,
  S9 DTM method, and export validation.
- Run the max profile with the Apache DA3/gsplat path on a high-memory CUDA host and compare against the
  balanced TSDF result using held-out checkpoints.
- Add a real SAM2+tracker path only after its licenses/checkpoints are pinned and its masks are consumed by
  S4/S7.
- Implement UV texture generation and validate re-opened textured GLB/OBJ artifacts.
