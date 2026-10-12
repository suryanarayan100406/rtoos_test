"""S7 failure evidence survives native errors and keeps fallback grids distinct."""
import numpy as np
import pytest

from drishti.bundle import Bundle
from drishti.config import DrishtiConfig
from drishti.io.pointcloud import read_ply_points
from drishti.stages.s7_dense import DenseStage


@pytest.fixture
def dense_bundle(tmp_path):
    bundle = Bundle.create(tmp_path, run_id="dense-evidence")
    bundle.write_json("s6_global/poses_global.json", {
        "intrinsics": {"model": "PINHOLE", "params": [100, 100, 2, 2], "width": 4, "height": 4},
        "poses": [{"name": "frame.jpg", "R": np.eye(3).tolist(), "center_crs": [0, 0, 10]}],
    })
    bundle.write_json("s6_global/views.json", {"keyframes": [{"path": "frame.jpg", "index": 0}]})
    bundle.write_json("s6_global/sparse_points.json", {"points": []})
    bundle.write_json("s4_depth/depth.json", {
        "n_frames": 1, "n_metric_aligned": 1,
        "frames": [{"index": 0, "metric": True, "depth": "s4_depth/0.npy"}],
    })
    np.save(bundle.artifact_path("s4_depth/0.npy"), np.ones((4, 4), np.float32))
    return bundle


def _event(backend, emitted=0):
    return {
        "tile_id": "0:0", "fusion_backend": backend,
        "candidate_frames": 1, "frames_integrated": 1,
        "points_extracted": 2, "points_emitted": emitted,
        "geometry_class": "reconstructed" if emitted else "unknown",
        "status": "reconstructed" if emitted else "unobservable",
        "reason": None if emitted else "no_surface_in_core",
        "z_band": [-float("inf"), float("inf")],
    }


@pytest.mark.parametrize("native_error", [False, True])
def test_failed_fusion_persists_diagnostics(dense_bundle, monkeypatch, native_error):
    bundle = dense_bundle

    def fuse(*args, tile_observer, sink, **kwargs):
        tile_observer(_event("adaptive_metric"))
        # Evidence is already on disk while native fusion is still running.
        progress = bundle.read_json("s7_dense/tile_states.json")
        assert progress["status"] == "running"
        assert progress["tiles"][0]["points_extracted"] == 2
        assert progress["tiles"][0]["z_band"] == [None, None]
        if native_error:
            sink(np.zeros((1, 3)), np.ones((1, 3)))
            raise RuntimeError("native interrupted")
        return np.empty((0, 3)), np.empty((0, 3)), 1

    monkeypatch.setattr("drishti.recon.tsdf.fuse_tsdf_tiled", fuse)
    message = "native interrupted" if native_error else "points_extracted=2, points_emitted=0"
    with pytest.raises(RuntimeError, match=message):
        DenseStage().run(bundle, DrishtiConfig(), None)
    evidence = bundle.read_json("s7_dense/tile_states.json")
    assert evidence["status"] == "failed"
    assert message in evidence["error"]
    assert len(evidence["attempts"]["adaptive_metric"]) == 1
    assert not bundle.exists("s7_dense/dense.json")
    assert not bundle.exists("s7_dense/dense.ply")
    assert not bundle.exists("s7_dense/dense.ply.body.tmp")


def test_fallback_keeps_failed_evidence_and_reports_degradation(dense_bundle, monkeypatch):
    bundle = dense_bundle

    def fuse(*args, tile_observer, sink, **kwargs):
        tile_observer(_event("adaptive_metric"))
        # A second failed adaptive tile must not count as a gap in the fallback's different grid.
        tile_observer({**_event("adaptive_metric"), "tile_id": "1:0"})
        sink(np.array([[1.0, 2.0, 3.0]]), np.ones((1, 3)))
        tile_observer({**_event("legacy_center", emitted=1), "fallback_reason": "adaptive_empty_cloud"})
        return np.empty((0, 3)), np.empty((0, 3)), 1

    monkeypatch.setattr("drishti.recon.tsdf.fuse_tsdf_tiled", fuse)
    result = DenseStage().run(bundle, DrishtiConfig(), None)
    evidence = bundle.read_json(result.confidence_ref)
    assert evidence["status"] == "complete"
    assert len(evidence["attempts"]["adaptive_metric"]) == 2
    assert evidence["attempts"]["adaptive_metric"][0]["points_emitted"] == 0
    assert len(evidence["tiles"]) == 1
    assert evidence["tiles"][0]["fusion_backend"] == "legacy_center"
    assert result.metrics["n_reconstructed_tiles"] == 1
    assert result.metrics["n_unknown_tiles"] == 0
    assert result.metrics["fallback_reason"] == "adaptive_empty_cloud"
    assert "legacy camera-center" in result.degraded_reason
    dense = bundle.read_json(result.outputs["dense"])
    assert dense["fusion_backend"] == "legacy_center"
    assert not dense["measurement_authoritative"]
    points, _ = read_ply_points(bundle.artifact_path(result.outputs["cloud"]))
    np.testing.assert_array_equal(points, [[1.0, 2.0, 3.0]])
