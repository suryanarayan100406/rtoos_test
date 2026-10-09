"""Accuracy + confidence report builder (pure, testable).

Summarizes a finished run STRICTLY from measured manifest data: per-stage status, wall time, confidence,
degradation reasons, and key metrics. Accuracy is reported as the S2 spine fit RMSE and, only if the
user supplied ground check-points, a validated error; otherwise it is labelled "unvalidated". Nothing
here invents numbers (AGENTS.md §5, §10 honesty policy).
"""
from __future__ import annotations

import html


def build_report(manifest: dict, checkpoints: dict | None = None) -> dict:
    """Assemble the report dict from a manifest dict (+ optional checkpoint validation results)."""
    stages = manifest.get("stages", [])
    total_wall = sum((s.get("wall_seconds") or 0.0) for s in stages)

    stage_rows = []
    for s in stages:
        stage_rows.append({
            "name": s.get("name"),
            "status": s.get("status"),
            "wall_seconds": round(s.get("wall_seconds") or 0.0, 2),
            "confidence": (s.get("confidence") or {}).get("summary"),
            "degraded_reason": s.get("degraded_reason"),
            "environment": s.get("environment"),
            "metrics": s.get("metrics", {}),
            "outputs": s.get("outputs", {}),
        })

    def _metric(stage_name, key):
        for s in stages:
            if s.get("name") == stage_name:
                return (s.get("metrics") or {}).get(key)
        return None

    spine_rmse = _metric("s2_poses", "spine_fit_rmse_m")
    reconstructed_tiles = _metric("s7_dense", "n_reconstructed_tiles")
    unknown_tiles = _metric("s7_dense", "n_unknown_tiles")
    dense_method = _metric("s7_dense", "method")
    measurement_authoritative = _metric("s7_dense", "measurement_authoritative")
    accuracy = {
        "spine_fit_rmse_m": spine_rmse,
        "spine_fit_label": "GPS-alignment fit residual (design proxy, not an independent check)",
    }
    if checkpoints:
        accuracy["checkpoint"] = checkpoints  # {n, rmse_z_m, mae_z_m, note}
        accuracy["validated"] = True
    else:
        accuracy["validated"] = False
        accuracy["note"] = "No ground check-points supplied — absolute accuracy is UNVALIDATED."

    n_failed = sum(1 for s in stages if s.get("status") == "failed")
    n_degraded = sum(1 for s in stages if s.get("degraded_reason"))
    return {
        "run_id": manifest.get("run_id"),
        "dataset": manifest.get("dataset_name"),
        "drishti_version": manifest.get("drishti_version"),
        "crs": manifest.get("crs"),
        "total_wall_seconds": round(total_wall, 2),
        "n_stages": len(stages),
        "n_failed": n_failed,
        "n_degraded": n_degraded,
        "accuracy": accuracy,
        "geometry_evidence": {
            "classes": ["observed", "reconstructed", "inferred", "unknown"],
            "dense_method": dense_method,
            "reconstructed_tiles": reconstructed_tiles,
            "unknown_tiles": unknown_tiles,
            "inferred_tiles": None,
            "measurement_authoritative": measurement_authoritative,
            "note": "Per-element evidence is in s7_dense/tile_states.json; null means no measured count was emitted.",
        },
        "ai_usage": {
            "depth_model": _metric("s4_depth", "model"),
            "dense_method": dense_method,
            "depth_metric_aligned_frame_fraction": _metric("s4_depth", "frac_metric_aligned"),
            "dynamic_mask_fraction": _metric("s3_masking", "mean_dynamic_frac"),
        },
        "stages": stage_rows,
    }


def render_html(report: dict) -> str:
    """Render the report dict as a self-contained HTML page (no external assets)."""
    e = lambda x: html.escape(str(x))  # noqa: E731
    crs = report.get("crs") or {}
    acc = report.get("accuracy", {})
    rows = []
    for s in report["stages"]:
        conf = s["confidence"]
        conf_txt = "—" if conf is None else f"{conf:.3f}"
        deg = f"<div class='deg'>⚠ {e(s['degraded_reason'])}</div>" if s["degraded_reason"] else ""
        status_cls = {"done": "ok", "degraded": "warn", "failed": "err"}.get(s["status"], "")
        rows.append(
            f"<tr><td>{e(s['name'])}</td>"
            f"<td class='{status_cls}'>{e(s['status'])}</td>"
            f"<td>{s['wall_seconds']:.2f}s</td>"
            f"<td>{conf_txt}</td>"
            f"<td>{e(s.get('environment') or '')}{deg}</td></tr>"
        )
    acc_line = (
        f"Validated vs {acc['checkpoint']['n']} check-points: RMSE_z = "
        f"{acc['checkpoint'].get('rmse_z_m')} m"
        if acc.get("validated") else f"<b>UNVALIDATED</b> — {e(acc.get('note',''))}"
    )
    spine = acc.get("spine_fit_rmse_m")
    spine_txt = "n/a" if spine is None else f"{spine} m"
    evidence = report.get("geometry_evidence", {})
    ai = report.get("ai_usage", {})
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>DRISHTI report — {e(report['run_id'])}</title>
<style>
 body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2rem;color:#111;background:#fafafa}}
 h1{{margin-bottom:0}} .sub{{color:#666}} table{{border-collapse:collapse;width:100%;margin-top:1rem}}
 th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px;vertical-align:top}}
 th{{background:#f0f0f0}} .ok{{color:#137333;font-weight:600}} .warn{{color:#b06000;font-weight:600}}
 .err{{color:#b00020;font-weight:600}} .deg{{color:#b06000;font-size:12px;margin-top:3px}}
 .card{{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;margin-top:1rem}}
 code{{background:#eee;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>DRISHTI reconstruction report</h1>
<div class="sub">run <code>{e(report['run_id'])}</code> · dataset <b>{e(report['dataset'])}</b>
 · v{e(report['drishti_version'])}</div>
<div class="card">
 <b>CRS:</b> EPSG:{e(crs.get('epsg'))} ({e(crs.get('derived_from'))}, {e(crs.get('vertical'))})<br>
 <b>Total wall time:</b> {report['total_wall_seconds']}s ·
 <b>Stages:</b> {report['n_stages']} ({report['n_failed']} failed, {report['n_degraded']} degraded)<br>
 <b>Spine fit RMSE:</b> {spine_txt} <span class="sub">({e(acc.get('spine_fit_label',''))})</span><br>
 <b>Absolute accuracy:</b> {acc_line}<br>
 <b>Geometry evidence:</b> {e(evidence.get('reconstructed_tiles'))} reconstructed tiles ·
 {e(evidence.get('unknown_tiles'))} unknown/unobservable tiles · measurement-authoritative =
 {e(evidence.get('measurement_authoritative'))}<br>
 <b>AI usage:</b> depth={e(ai.get('depth_model'))} · dense={e(ai.get('dense_method'))} ·
 metric-aligned-frame-fraction={e(ai.get('depth_metric_aligned_frame_fraction'))}
</div>
<table><tr><th>Stage</th><th>Status</th><th>Wall</th><th>Confidence</th><th>Environment / notes</th></tr>
{''.join(rows)}
</table>
<p class="sub">Confidence is a measured, per-stage self-assessment (0–1). Degradation notes mark where a
real fallback was used. This report contains only measured values.</p>
</body></html>"""
