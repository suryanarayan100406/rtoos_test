"""Report builder + HTML renderer (measured-only, honest accuracy labelling)."""
import json

from drishti.report import build_report, render_html


def _manifest() -> dict:
    return {
        "run_id": "r1",
        "dataset_name": "demo",
        "drishti_version": "0.1.0",
        "crs": {"epsg": 32643, "derived_from": "gps_median_lonlat", "vertical": "ellipsoidal"},
        "stages": [
            {"name": "s0_ingest", "status": "done", "wall_seconds": 1.5,
             "confidence": {"summary": 0.9}, "metrics": {}, "environment": "local"},
            {"name": "s2_poses", "status": "done", "wall_seconds": 10.0,
             "confidence": {"summary": 0.8}, "metrics": {"spine_fit_rmse_m": 0.42},
             "environment": "cloud-t4"},
            {"name": "s3_masking", "status": "degraded", "wall_seconds": 2.0,
             "confidence": {"summary": 0.5}, "degraded_reason": "SAM2 refinement not enabled",
             "metrics": {}, "environment": "local"},
        ],
    }


def test_build_report_measured_only():
    r = build_report(_manifest())
    assert r["total_wall_seconds"] == 13.5
    assert r["n_stages"] == 3
    assert r["n_failed"] == 0
    assert r["n_degraded"] == 1
    assert r["accuracy"]["spine_fit_rmse_m"] == 0.42
    assert r["accuracy"]["validated"] is False           # no check-points supplied


def test_build_report_with_checkpoints():
    r = build_report(_manifest(), checkpoints={"n": 5, "rmse_z_m": 0.3, "mae_z_m": 0.2})
    assert r["accuracy"]["validated"] is True
    assert r["accuracy"]["checkpoint"]["n"] == 5


def test_render_html_unvalidated_contains_fields():
    html = render_html(build_report(_manifest()))
    assert "<html" in html.lower()
    assert "r1" in html                                  # run id
    assert "demo" in html                                # dataset
    assert "UNVALIDATED" in html
    assert "s3_masking" in html
    assert "SAM2 refinement not enabled" in html


def test_render_html_validated_reports_checkpoints():
    r = build_report(_manifest(), checkpoints={"n": 5, "rmse_z_m": 0.3, "mae_z_m": 0.2})
    html = render_html(r)
    assert "check-points" in html
    assert "0.3" in html


def test_checkpoint_rows_accept_csv_and_geojson(tmp_path):
    from drishti.stages.report import _checkpoint_rows

    csv_path = tmp_path / "checkpoints.csv"
    csv_path.write_text("latitude,longitude,elevation\n44.1,-72.2,250.5\n", encoding="utf-8")
    assert _checkpoint_rows(csv_path) == [(44.1, -72.2, 250.5)]

    geojson_path = tmp_path / "checkpoints.geojson"
    geojson_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-72.2, 44.1, 250.5]},
            "properties": {},
        }],
    }), encoding="utf-8")
    assert _checkpoint_rows(geojson_path) == [(44.1, -72.2, 250.5)]
