"""API server smoke tests. Skipped unless the [server] extra (FastAPI + httpx) is installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from server.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_doctor_reports_compute_and_stage_readiness():
    r = client.get("/api/doctor")
    assert r.status_code == 200
    d = r.json()
    assert "cpu_count" in d["compute"]
    assert "cuda" in d["compute"]
    # S6 needs OpenCV/pyproj for rectified views and CRS transforms; readiness follows the shared CLI policy.
    from drishti.cli import _stage_readiness

    assert d["stages"]["s6_global"]["ready"] is _stage_readiness()[0]["s6_global"][0]
    assert d["stages"]["report"]["ready"] is True
    assert d["pipeline_order"][0] == "s0_ingest"


def test_list_runs_ok_when_empty(tmp_path, monkeypatch):
    # point the server at an empty runs dir for this test
    import server.app as srv
    monkeypatch.setattr(srv, "RUNS_DIR", tmp_path)
    r = client.get("/api/runs")
    assert r.status_code == 200
    assert r.json() == {"runs": []}


def test_start_run_requires_a_dataset():
    r = client.post("/api/runs", json={})
    assert r.status_code == 422


def test_start_run_missing_dataset_file_is_404():
    r = client.post("/api/runs", json={"dataset_path": "does_not_exist.yaml"})
    assert r.status_code == 404


def test_get_missing_run_is_404():
    r = client.get("/api/runs/no_such_run")
    assert r.status_code == 404


def test_report_missing_is_404():
    r = client.get("/api/runs/no_such_run/report")
    assert r.status_code == 404


def test_artifact_missing_is_404():
    r = client.get("/api/runs/no_such_run/files/mesh.glb")
    assert r.status_code == 404
