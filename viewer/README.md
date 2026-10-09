# DRISHTI web viewer

A React + Vite + TypeScript single-page app that inspects pipeline runs and renders the
**georeferenced 3D output** in the browser with **CesiumJS**.

It is a thin, honest window onto the API (`server/app.py`): every number it shows — stage status,
where each stage ran (`local` / `cloud-t4`), wall-clock seconds, per-stage confidence, degradation
reasons — comes straight from the on-disk `manifest.json` the runner writes. Nothing is simulated.

## Stack & why

- **Vite + React + TypeScript** — fast dev server, typed components.
- **CesiumJS** — renders the **3D Tiles** tileset (the georeferenced mesh from `s10_export`) on a
  globe at its real-world coordinates. Configured **fully offline**: no Cesium Ion token, no external
  imagery/terrain, no CDN. Cesium's static assets are copied into the build by
  `vite-plugin-static-copy`, and `window.CESIUM_BASE_URL` (set in `index.html`) points at them.

### About point clouds / Potree

The pipeline can export a **Potree** point cloud (guarded by `PotreeConverter` being on PATH). Potree
is a *separate* WebGL renderer whose build must be vendored to embed it, and its format is not what
Cesium loads — so this viewer does **not** fake an in-browser point-cloud render. When a run produces
Potree output it appears as a download in the **Deliverables** card. Cesium already renders the
georeferenced *mesh* (3D Tiles), which covers the primary "view the model" need. Embedding Potree is a
documented, optional future addition (vendor the Potree build under `public/potree/` and add a panel).

## Develop

The viewer talks to the API over same-origin relative URLs; in dev, Vite proxies `/api` → `:8000`.

```bash
# 1) start the API (from the repo root, in another terminal)
pip install -e ".[server]"
DRISHTI_RUNS_DIR=runs python -m server.app         # or: uvicorn server.app:app --port 8000

# 2) start the viewer
cd viewer
npm install
npm run dev                                         # http://localhost:5173
```

`npm run typecheck` runs the full TypeScript check (the build transpiles without type-checking for
speed; CI/typecheck is where types are enforced).

## Build for production

```bash
cd viewer
npm ci
npm run build            # -> viewer/dist (self-contained: includes Cesium assets under dist/cesium)
```

When `viewer/dist` exists, the API server mounts it at `/`, so the whole thing is served from one
origin (no CORS, no proxy) — see `server/app.py`. In Docker, build `dist` and mount it into the image
at `/app/viewer/dist` (see `docker/README.md`).

## What it shows

- **Runs** (left) — every bundle under `DRISHTI_RUNS_DIR`, with rollup state and stage counts; polls
  every 5 s so a running pipeline updates live.
- **Run detail** — the per-stage table (status / where / seconds / confidence / degradation), a link
  to the accuracy report, the Cesium 3D view, and a Deliverables list linking each exported format
  (skipped formats show their reason).
- **Doctor** — the same measured hardware + stage-readiness the CLI `doctor` prints.
