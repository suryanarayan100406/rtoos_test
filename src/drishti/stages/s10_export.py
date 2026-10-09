"""S10 — Export deliverables.

Emits the required output formats (OBJ, PLY, LAS, GeoTIFF, glTF/GLB, FBX; plus LAZ, 3D Tiles, Potree
when their tools are present) from the canonical bundle products. Each format is attempted with real
tools; a format whose tool is missing is recorded as skipped-with-reason, never faked. If a MUST format
is skipped the stage is marked degraded so the report surfaces it.
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage

_MUST = {"obj", "ply", "las", "geotiff", "gltf", "glb", "fbx"}


@register_stage
class ExportStage(Stage):
    name = "s10_export"
    requires = ("s8_mesh", "s9_geo")
    compute = ComputeNeed(cpu_only=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"export": cfg.export.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import json
        from pathlib import Path

        from ..io import exporters as ex

        mesh_doc = bundle.read_json("s8_mesh/mesh.json")
        dense_doc = bundle.read_json("s7_dense/dense.json")
        geo_doc = bundle.read_json("s9_geo/geo.json")
        if bundle.manifest.crs is None or bundle.manifest.crs.epsg is None:
            raise RuntimeError("s10_export: manifest CRS is missing")
        epsg = bundle.manifest.crs.epsg

        mesh_ply = Path(bundle.artifact_path(mesh_doc["mesh"]))
        dense_ply = Path(bundle.artifact_path(dense_doc["cloud"]))
        out = bundle.stage_dir(self.name)

        results: dict[str, dict[str, Any]] = {}
        artifact_outputs: dict[str, str] = {}

        def record(fmt: str, ok: bool, info: str):
            if ok:
                ok, info = ex.validate_export(fmt, Path(info))
            results[fmt] = {"ok": ok, "path": bundle.relpath(info) if ok else None,
                            "reason": None if ok else info}
            if ok:
                artifact_outputs[f"export_{fmt}"] = bundle.relpath(info)

        wanted = [f.lower() for f in (cfg.export.formats + cfg.export.also)]
        for fmt in wanted:
            if fmt == "obj":
                record(fmt, *ex.export_obj(mesh_ply, out / "model.obj"))
            elif fmt == "ply":
                record(fmt, *ex.copy_product(mesh_ply, out / "model.ply"))
            elif fmt in ("gltf", "glb"):
                textured_asset = mesh_doc.get("textured_asset")
                if fmt == "glb" and textured_asset:
                    record(fmt, *ex.copy_product(bundle.artifact_path(textured_asset), out / "model.glb"))
                    if results[fmt]["ok"] and mesh_doc.get("texture_origin_crs") is not None:
                        georef = out / "model.glb.georef.json"
                        georef.write_text(json.dumps({
                            "origin_crs": mesh_doc["texture_origin_crs"],
                            "coordinate_frame": "local_crs_translation",
                        }, indent=2), encoding="utf-8")
                        results[fmt]["georef_sidecar"] = bundle.relpath(georef)
                        artifact_outputs["export_glb_georef"] = bundle.relpath(georef)
                else:
                    record(fmt, *ex.export_glb(mesh_ply, out / f"model.{fmt}"))
            elif fmt == "fbx":
                source = bundle.artifact_path(mesh_doc.get("textured_asset", mesh_doc["mesh"]))
                record(fmt, *ex.export_fbx_via_blender(source, out / "model.fbx"))
            elif fmt == "las":
                record(fmt, *ex.export_las_from_ply(dense_ply, out / "points.las", epsg, cfg.export.point_format))
            elif fmt == "laz":
                record(fmt, *ex.export_las_from_ply(dense_ply, out / "points.laz", epsg, cfg.export.point_format, laz=True))
            elif fmt == "geotiff":
                # copy DSM/DTM/ortho
                any_ok = False
                geo_out = out / "geotiff"
                for key in ("dsm", "dtm", "ortho"):
                    if key in geo_doc and geo_doc[key]:
                        ok, info = ex.copy_product(bundle.artifact_path(geo_doc[key]), geo_out / f"{key}.tif")
                        any_ok = any_ok or ok
                record("geotiff", any_ok, str(geo_out) if any_ok else "no GeoTIFFs from S9")
            elif fmt == "3dtiles":
                record(fmt, *ex.export_3dtiles(mesh_ply, out / "3dtiles"))
            elif fmt == "potree":
                record(fmt, *ex.export_potree(dense_ply, out / "potree"))
            else:
                record(fmt, False, f"unknown format '{fmt}'")

        must_wanted = [f for f in wanted if f in _MUST]
        must_ok = [f for f in must_wanted if results.get(f, {}).get("ok")]
        skipped_must = [f for f in must_wanted if not results.get(f, {}).get("ok")]

        export_ref = bundle.write_json(f"{self.name}/exports.json", {
            "epsg": epsg, "results": results,
            "must_formats": must_wanted, "must_ok": must_ok, "skipped_must": skipped_must,
        })
        outputs = {"exports": export_ref, **artifact_outputs}

        degraded = None
        if skipped_must:
            degraded = "MUST formats skipped (missing tools): " + ", ".join(skipped_must)
        return StageResult(
            outputs=outputs,
            metrics={
                "formats_ok": [f for f, r in results.items() if r["ok"]],
                "formats_skipped": {f: r["reason"] for f, r in results.items() if not r["ok"]},
                "n_ok": sum(1 for r in results.values() if r["ok"]),
            },
            confidence_summary=round(len(must_ok) / max(1, len(must_wanted)), 3),
            degraded_reason=degraded,
        )
