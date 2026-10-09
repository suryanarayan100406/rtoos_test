"""The stage runner: orders stages by dependency, skips fresh ones (content-hash based), places each
local/cloud, times it, records confidence + measured metrics, and persists the manifest after every
stage so a run is always resumable. Unhandled stage errors are recorded and re-raised (fail loudly);
a stage that used a fallback returns a *degraded* result rather than raising.
"""
from __future__ import annotations

import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from ..bundle import Bundle, StageRecord, combine, sha256_file
from ..config import DrishtiConfig
from ..logging import get_logger
from ..runtime import Runtime
from .base import Stage, get_stage

log = get_logger("drishti.runner")

# Canonical offline pipeline order (S5 live-fusion is folded into s7_dense; see AGENTS.md §2).
PIPELINE_ORDER = [
    "s0_ingest",
    "s1_frameqa",
    "s2_poses",
    "s6_global",
    "s3_masking",
    "s4_depth",
    "s7_dense",
    "s8_mesh",
    "s9_geo",
    "s10_export",
    "report",
]


class StageFailed(RuntimeError):
    pass


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _hash_output_path(p: Path) -> str:
    if p.is_file():
        return sha256_file(p)
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file())
        return combine(*[sha256_file(f) for f in files]) if files else ""
    return ""


def _compute_inputs_hash(stage: Stage, bundle: Bundle, cfg: DrishtiConfig) -> str:
    parts: list[str] = []
    for dep in stage.requires:
        rec = bundle.manifest.stage(dep)
        if rec is None or rec.outputs_hash is None:
            # dependency not yet produced — freshness cannot match; force a (re)run
            parts.append(f"missing:{dep}")
        else:
            parts.append(rec.outputs_hash)
    parts.extend(stage.extra_input_hashes(bundle, cfg))
    return combine(*parts)


def resolve_order(only: str | None, upto: str | None) -> list[str]:
    if only:
        if only not in PIPELINE_ORDER:
            raise KeyError(f"unknown stage: {only}")
        return [only]
    if upto:
        if upto not in PIPELINE_ORDER:
            raise KeyError(f"unknown stage: {upto}")
        return PIPELINE_ORDER[: PIPELINE_ORDER.index(upto) + 1]
    return list(PIPELINE_ORDER)


def run_pipeline(
    bundle: Bundle,
    cfg: DrishtiConfig,
    rt: Runtime,
    only: str | None = None,
    upto: str | None = None,
    force: bool = False,
) -> None:
    """Run the (subset of the) pipeline over ``bundle``. Persists the manifest after each stage."""
    order = resolve_order(only, upto)
    from ..bundle.hashing import sha256_obj  # local import to avoid cycle at module load

    for name in order:
        stage = get_stage(name)

        # verify dependencies are satisfied when running a single stage
        for dep in stage.requires:
            rec = bundle.manifest.stage(dep)
            if rec is None or rec.status not in ("done", "degraded"):
                raise StageFailed(
                    f"cannot run '{name}': dependency '{dep}' has not completed. "
                    f"Run it first (e.g. `drishti resume`) or run the full pipeline."
                )

        params = stage.params(cfg)
        params_hash = sha256_obj(params)
        inputs_hash = _compute_inputs_hash(stage, bundle, cfg)
        placement = rt.place(stage.compute)

        prev = bundle.manifest.stage(name)
        fresh = (
            not force
            and prev is not None
            and prev.status in ("done", "degraded")
            and prev.inputs_hash == inputs_hash
            and prev.params_hash == params_hash
        )
        if fresh:
            log.info("· %s — fresh, skipping", name)
            continue

        record = StageRecord(
            name=name,
            status="running",
            environment=placement,
            started_utc=_utc(),
            inputs_hash=inputs_hash,
            params=params,
            params_hash=params_hash,
        )
        bundle.manifest.upsert_stage(record)
        bundle.save_manifest()

        log.info("▶ %s  [%s]", name, placement)
        t0 = time.perf_counter()
        try:
            result = stage.run(bundle, cfg, rt)
        except Exception as exc:  # record the real failure, then re-raise (no faked success)
            record.status = "failed"
            record.ended_utc = _utc()
            record.wall_seconds = round(time.perf_counter() - t0, 3)
            record.error = f"{type(exc).__name__}: {exc}"
            bundle.manifest.upsert_stage(record)
            bundle.save_manifest()
            log.error("✖ %s failed after %.1fs: %s", name, record.wall_seconds, exc)
            log.debug("%s", traceback.format_exc())
            raise StageFailed(f"stage '{name}' failed: {exc}") from exc

        wall = round(time.perf_counter() - t0, 3)
        # compute outputs hash from produced files/dirs
        out_hashes = [
            _hash_output_path(bundle.artifact_path(p)) for p in result.outputs.values()
        ]
        record.status = "degraded" if result.degraded_reason else "done"
        record.ended_utc = _utc()
        record.wall_seconds = wall
        record.outputs = result.outputs
        record.outputs_hash = combine(*out_hashes) if out_hashes else ""
        record.metrics = result.metrics
        record.confidence.summary = result.confidence_summary
        record.confidence.detail_ref = result.confidence_ref
        record.degraded_reason = result.degraded_reason
        bundle.manifest.upsert_stage(record)
        bundle.save_manifest()

        tag = "degraded" if result.degraded_reason else "done"
        log.info("✓ %s — %s in %.1fs", name, tag, wall)
