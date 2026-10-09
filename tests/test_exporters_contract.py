import json

import numpy as np
import pytest


def test_glb_export_recenters_large_crs_coordinates_and_records_origin(tmp_path):
    pytest.importorskip("trimesh")
    from drishti.io.exporters import export_glb, validate_export
    from drishti.io.pointcloud import write_ply_points

    points = np.array([
        [437021.0, 4572795.0, 248.0],
        [437021.5, 4572795.0, 248.0],
        [437021.0, 4572795.5, 248.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    source = tmp_path / "mesh.ply"
    write_ply_points(source, points)
    # PLY point cloud is not a triangle mesh, so use trimesh to create the source mesh directly.
    import trimesh

    trimesh.Trimesh(vertices=points, faces=faces, process=False).export(source)
    dest = tmp_path / "model.glb"
    ok, result = export_glb(source, dest)
    assert ok and result == str(dest)
    assert validate_export("glb", dest)[0]
    sidecar = json.loads(dest.with_suffix(dest.suffix + ".georef.json").read_text())
    assert np.allclose(sidecar["origin_crs"], points.mean(axis=0))

    loaded = trimesh.load(dest, process=False, force="mesh")
    assert np.max(np.abs(np.asarray(loaded.vertices))) < 2.0
