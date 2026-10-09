from pathlib import Path


def test_geotiff_export_isolated_from_later_s10_artifacts(tmp_path, monkeypatch):
    from drishti.bundle import Bundle
    from drishti.bundle.manifest import CrsInfo
    from drishti.config import DrishtiConfig
    from drishti.stages.s10_export import ExportStage

    bundle = Bundle.create(tmp_path, run_id="run")
    bundle.manifest.crs = CrsInfo(epsg=32643)
    mesh = bundle.stage_dir("s8_mesh") / "mesh.ply"
    dense = bundle.stage_dir("s7_dense") / "dense.ply"
    dsm = bundle.stage_dir("s9_geo") / "dsm.tif"
    for path in (mesh, dense, dsm):
        path.write_bytes(b"artifact")
    bundle.write_json("s8_mesh/mesh.json", {"mesh": bundle.relpath(mesh)})
    bundle.write_json("s7_dense/dense.json", {"cloud": bundle.relpath(dense)})
    bundle.write_json("s9_geo/geo.json", {"dsm": bundle.relpath(dsm)})

    import drishti.io.exporters as exporters

    monkeypatch.setattr(exporters, "validate_export", lambda fmt, path: (True, str(path)))
    cfg = DrishtiConfig(export={"formats": ["geotiff"], "also": []})
    result = ExportStage().run(bundle, cfg, None)

    assert result.outputs["export_geotiff"] == "s10_export/geotiff"
    assert result.outputs["export_geotiff"] != "s10_export"
    assert Path(bundle.artifact_path(result.outputs["exports"])).is_file()
