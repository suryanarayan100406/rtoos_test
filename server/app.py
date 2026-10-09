"""DRISHTI API server.

A thin, honest HTTP layer over the exact same core the CLI uses: it starts real pipeline runs in a
background thread and reports status by reading the on-disk manifest (which the runner persists after
every stage), so status is always truthful and survives a server restart. It serves bundle artifacts
(GLB / 3D Tiles / point clouds / GeoTIFFs / the accuracy report) to the web viewer, with path-traversal
protection, and exposes the same hardware/dependency doctor the CLI does.

Nothing here fabricates progress or results. A run that cannot proceed (missing dependency, missing
input) fails loudly; its manifest records the failed stage and the API surfaces it verbatim.

Run it:

    pip install -e ".[server]"
    uvicorn server.app:app --reload            # dev
    DRISHTI_RUNS_DIR=/data/runs uvicorn server.app:app --host 0.0.0.0 --port 8000

Environment:
    DRISHTI_RUNS_DIR     where bundles live / are created   (default: ./runs)
    DRISHTI_CONFIG_DIR   config tree for profiles/datasets   (default: ./configs)
    DRISHTI_CORS_ORIGINS comma-separated allowed origins     (default: http://localhost:5173)
"""
from __future__ import annotations

import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from drishti import __version__
from drishti.bundle import Bundle, new_run_id

# Readiness/doctor logic lives with the CLI; reuse it rather than duplicate the policy.
from drishti.cli import _stage_readiness
from drishti.config import load_config
from drishti.config.loader import ConfigError
from drishti.runtime import Runtime, detect_compute
from drishti.stages import PIPELINE_ORDER, resolve_order, run_pipeline

RUNS_DIR = Path(os.environ.get("DRISHTI_RUNS_DIR", "runs")).resolve()
CONFIG_DIR = Path(os.environ.get("DRISHTI_CONFIG_DIR", "configs")).resolve()
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("DRISHTI_CORS_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]
VIEWER_DIST = Path(__file__).resolve().parent.parent / "viewer" / "dist"

app = FastAPI(title="DRISHTI API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory record of runs THIS process launched. The manifest on disk is the source of truth; this
# only adds the launch error (if the thread died before writing a manifest) and a liveness flag.
_LAUNCHED: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- models
class RunRequest(BaseModel):
    dataset_path: str | None = Field(
        default=None, description="Path (server-side) to a dataset descriptor YAML."
    )
    dataset: dict[str, Any] | None = Field(
        default=None, description="Inline dataset descriptor (written to a temp YAML if given)."
    )
    profile: str | None = Field(default=None, description="Profile name (fast/balanced/max).")
    set: list[str] = Field(default_factory=list, description="Dotted config overrides key=value.")
    only: str | None = None
    upto: str | None = None
    force: bool = False


# --------------------------------------------------------------------------- helpers
def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _rollup(manifest: dict[str, Any]) -> str:
    """Overall run state derived purely from stage records — never guessed."""
    stages = {s["name"]: s["status"] for s in manifest.get("stages", [])}
    if any(v == "failed" for v in stages.values()):
        return "failed"
    if any(v == "running" for v in stages.values()):
        return "running"
    if not stages:
        return "empty"
    done = {n for n, v in stages.items() if v in ("done", "degraded")}
    if set(PIPELINE_ORDER).issubset(done):
        return "done"
    return "partial"


def _run_dir(run_id: str) -> Path:
    d = (RUNS_DIR / run_id).resolve()
    # never allow a run_id to escape RUNS_DIR
    if RUNS_DIR not in d.parents and d != RUNS_DIR:
        raise HTTPException(status_code=400, detail="invalid run id")
    return d


def _load_manifest(run_id: str) -> dict[str, Any]:
    d = _run_dir(run_id)
    mpath = d / "manifest.json"
    if not mpath.is_file():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    b = Bundle.open(d)
    return b.manifest.model_dump(mode="json")


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    stages = manifest.get("stages", [])
    return {
        "run_id": manifest.get("run_id"),
        "dataset_name": manifest.get("dataset_name"),
        "drishti_version": manifest.get("drishti_version"),
        "state": _rollup(manifest),
        "crs": manifest.get("crs"),
        "n_stages_done": sum(1 for s in stages if s["status"] in ("done", "degraded")),
        "n_stages_failed": sum(1 for s in stages if s["status"] == "failed"),
        "n_stages_total": len(PIPELINE_ORDER),
        "stages": [
            {
                "name": s["name"],
                "status": s["status"],
                "environment": s.get("environment"),
                "wall_seconds": s.get("wall_seconds"),
                "confidence": (s.get("confidence") or {}).get("summary"),
                "degraded_reason": s.get("degraded_reason"),
                "error": s.get("error"),
            }
            for s in stages
        ],
    }


def _execute_run(run_id: str, dataset_path: str, req: RunRequest) -> None:
    """Runs in a background thread. Any failure is recorded; the manifest is the durable record."""
    try:
        cfg, descriptor = load_config(
            profile=req.profile,
            dataset=dataset_path,
            set_overrides=req.set,
            config_dir=CONFIG_DIR,
        )
        if descriptor is None:
            raise ConfigError(f"dataset descriptor did not load: {dataset_path}")
        # validate stage selection before doing any work (fail fast, exit-2 semantics on the API)
        resolve_order(req.only, req.upto)

        bundle = Bundle.create(
            output_root=RUNS_DIR,
            run_id=run_id,
            dataset_name=descriptor.name,
            config_snapshot=cfg.model_dump(mode="json"),
            dataset_snapshot=descriptor.model_dump(mode="json"),
            seeds={"global": cfg.run.seed},
        )
        rt = Runtime.build(cfg)
        with _LOCK:
            _LAUNCHED[run_id]["state"] = "running"
        run_pipeline(bundle, cfg, rt, only=req.only, upto=req.upto, force=req.force)
        with _LOCK:
            _LAUNCHED[run_id]["state"] = "finished"
    except Exception as exc:  # record every failure honestly; the manifest is the durable record
        with _LOCK:
            _LAUNCHED[run_id]["state"] = "error"
            _LAUNCHED[run_id]["error"] = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- endpoints
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "runs_dir": str(RUNS_DIR)}


@app.get("/api/doctor")
def doctor() -> dict[str, Any]:
    detected = detect_compute()
    readiness, have = _stage_readiness()
    return {
        "compute": detected.as_dict(),
        "modules": have,
        "stages": {name: {"ready": r, "detail": d} for name, (r, d) in readiness.items()},
        "pipeline_order": PIPELINE_ORDER,
    }


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    if RUNS_DIR.is_dir():
        for d in sorted(RUNS_DIR.iterdir()):
            if (d / "manifest.json").is_file():
                try:
                    m = Bundle.open(d).manifest.model_dump(mode="json")
                    runs.append(_summary(m))
                except Exception as exc:  # noqa: BLE001 — a corrupt bundle shouldn't 500 the list
                    runs.append({"run_id": d.name, "state": "unreadable", "error": str(exc)})
    return {"runs": runs}


@app.post("/api/runs", status_code=202)
def start_run(req: RunRequest = Body(...)) -> JSONResponse:
    if not req.dataset_path and not req.dataset:
        raise HTTPException(status_code=422, detail="provide 'dataset_path' or 'dataset'")

    dataset_path = req.dataset_path
    if dataset_path is None:
        raise RuntimeError("dataset_path is required after inline dataset materialization")
    if req.dataset is not None:
        # persist an inline descriptor so the run is fully reproducible from disk
        incoming = RUNS_DIR / "_incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        p = incoming / f"{uuid.uuid4().hex}.yaml"
        p.write_text(yaml.safe_dump(req.dataset, sort_keys=False), encoding="utf-8")
        dataset_path = str(p)

    if not Path(dataset_path).is_file():
        raise HTTPException(status_code=404, detail=f"dataset not found: {dataset_path}")

    # validate stage names up front so a bad request is a 400, not a background error
    try:
        resolve_order(req.only, req.upto)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_id = new_run_id()
    with _LOCK:
        _LAUNCHED[run_id] = {"state": "starting", "started_utc": _utc(), "error": None}
    t = threading.Thread(target=_execute_run, args=(run_id, dataset_path, req), daemon=True)
    t.start()
    return JSONResponse({"run_id": run_id, "state": "starting"}, status_code=202)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    summary = _summary(_load_manifest(run_id))
    with _LOCK:
        launch = _LAUNCHED.get(run_id)
    if launch and launch.get("error"):
        summary["launch_error"] = launch["error"]
    return summary


@app.get("/api/runs/{run_id}/manifest")
def get_manifest(run_id: str) -> dict[str, Any]:
    return _load_manifest(run_id)


@app.get("/api/runs/{run_id}/report")
def get_report(run_id: str) -> FileResponse:
    html = _run_dir(run_id) / "report" / "report.html"
    if not html.is_file():
        raise HTTPException(status_code=404, detail="report not generated yet")
    return FileResponse(html, media_type="text/html")


@app.get("/api/runs/{run_id}/files/{path:path}")
def get_artifact(run_id: str, path: str) -> FileResponse:
    """Serve a bundle artifact. Resolves inside the bundle dir only (no path traversal)."""
    base = _run_dir(run_id)
    target = (base / path).resolve()
    if base not in target.parents:
        raise HTTPException(status_code=400, detail="path escapes bundle")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"no such artifact: {path}")
    return FileResponse(target)


# --------------------------------------------------------------------------- static viewer
# Serve the built SPA at / when it exists (production). In dev the viewer runs on Vite (5173) and
# talks to this API cross-origin (see CORS above).
if VIEWER_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(VIEWER_DIST), html=True), name="viewer")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host=os.environ.get("DRISHTI_HOST", "127.0.0.1"),
        port=int(os.environ.get("DRISHTI_PORT", "8000")),
        reload=bool(os.environ.get("DRISHTI_RELOAD")),
    )


if __name__ == "__main__":
    main()
